import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date
import math
import pandas_datareader as web
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ticker =
start_date =
end_date =
df1 = yf.download(ticker, start=start_date, end=end_date)
print(df1)

def get_data(ticker, start_date, end_date):

  df1= yf.download(ticker, start=start_date, end=end_date)
  print('Initial dataset has {} samples, and {} features.'.format(df1.shape[0], \
                                                              df1.shape[1]))
  print("*"*32, "df1.head()", "*"*32)
  print(df1.head())
  return df1

df1=get_data(ticker, start_date, end_date)

def feature_generation(df1):

  df1["MACDdiff"] = ta.trend.macd_diff(df1["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)

  df1["EMA200"] = ta.trend.ema_indicator(df1["Close"], window=200, fillna=False)

  df1["ATR"] = ta.volatility.average_true_range(df1["High"], df1["Low"], df1["Close"] , 14 , False)

  df1["RSI"] = ta.momentum.rsi(df1["Close"], 14, True)

  vix = yf.download("^VIX", start=start_date, end=end_date)
  vix = vix[['Close']]
  vix.columns = ['VIX']
  df1 = pd.concat([df1, vix], axis=1)

  usdx = yf.download("DX-Y.NYB",  start=start_date, end=end_date)
  usdx = usdx[['Close']]
  usdx.columns = ['USDX']
  df1 = pd.concat([df1, usdx], axis=1)

  tnx = yf.download("^TNX",  start=start_date, end=end_date)
  tnx = tnx[['Close']]
  tnx.columns = ['TNX']
  df1 = pd.concat([df1, tnx], axis=1)

  print('The dataset now has {} samples, and {} features.'.format(df1.shape[0], \
                                                              df1.shape[1]))
  print(df1.head())
  return df1

df1 =feature_generation(df1)
df1 = df1.dropna()
df1

df1 = df1.drop(['Adj Close', 'High', 'Low', 'USDX', 'TNX', 'MACDdiff' ], axis=1)
df1.head()
#Remove the unnecessary features based on the selected ticker

df1 = df1[[df1.columns[1], df1.columns[0]] + list(df1.columns[2:])]

lag_days = 1

for i in range(lag_days, 0, -1):
    df1[f'Close'] = df1['Close'].shift(-i)

for i in range(lag_days, 0, -1):
    df1[f'Open'] = df1['Open'].shift(-i)

df1.dropna(inplace=True)

def split_data(df1):
  training_size = int(len(df1)*0.7)
  test_size = int((len(df1)-training_size)/2)
  df1 = (np.array(df1).reshape(-1,df1.shape[1]))

  train_data=df1[0:training_size, :]
  valid_data =  df1[training_size:(training_size+test_size), :]
  test_data= df1[len(df1)-test_size: len(df1), :]

  print("Training Data Length:", len(train_data))
  print("Validation Data Length:",len(valid_data))
  print("Test Data Length:",len(test_data))

  return train_data, valid_data, test_data

train_data, valid_data, test_data=split_data(df1)

scalar = MinMaxScaler()
train_data_scaled = scalar.fit_transform(train_data)

def scaling_train_data (train_data, time_step):

  scalar = MinMaxScaler()
  train_data_scaled = scalar.fit_transform(train_data)
  X_train_scaled = train_data_scaled[:,1:(train_data_scaled.shape[1] + 1)]
  y_train_scaled = train_data_scaled[:,0:1]

  X_train = []
  y_train = []

  for i in range(time_step, train_data.shape[0]):
    X_train.append(X_train_scaled[i-time_step: i])
    y_train.append(y_train_scaled[i, 0])

  X_train, y_train = np.array(X_train), np.array(y_train)

  return  X_train , y_train

X_train , y_train = scaling_train_data (train_data,20)

def scaling_valid_data (valid_data, time_step):

  valid_data1 = pd.DataFrame(valid_data)
  train_data1 = pd.DataFrame(train_data)
  past = train_data1.tail(time_step)

  valid_data_tailed = past.append(valid_data1, ignore_index = True)
  inputs_valid = scalar.fit_transform(valid_data_tailed)

  X_valid = []
  y_valid = []

  X_valid_scaled = inputs_valid[:,1:(train_data_scaled.shape[1] + 1)]
  y_valid_scaled = inputs_valid[:,0:1]

  for i in range(time_step, inputs_valid.shape[0]):
    X_valid.append(X_valid_scaled[i-time_step:i])
    y_valid.append(y_valid_scaled[i, 0])

  X_valid, y_valid = np.array(X_valid), np.array(y_valid)

  return X_valid , y_valid

X_valid , y_valid = scaling_valid_data (valid_data, 20)

def scaling_test_data (test_data, time_step):

  test_data1 = pd.DataFrame(test_data)
  train_data1 = pd.DataFrame(train_data)
  past = train_data1.tail(time_step)

  test_data_tailed = past.append(test_data1, ignore_index = True)
  inputs = scalar.fit_transform(test_data_tailed)

  X_test = []
  y_test = []

  X_test_scaled = inputs[:,1:(train_data_scaled.shape[1] + 1)]
  y_test_scaled = inputs[:,0:1]

  for i in range(time_step, inputs.shape[0]):
    X_test.append(X_test_scaled[i-time_step:i])
    y_test.append(y_test_scaled[i, 0])

  X_test, y_test = np.array(X_test), np.array(y_test)

  return X_test , y_test

X_test , y_test = scaling_test_data (test_data, 20)

X_test.shape , y_test.shape, X_valid.shape, y_valid.shape, X_train.shape, y_train.shape

def model_generation (layer_size1, layer_size2, dense_layer, feature_number, activation_function):

  optimizer = Adam(learning_rate=0.01)

  model = Sequential()
  model.add(LSTM(layer_size1, activation = activation_function, return_sequences=True, input_shape =(X_train.shape[1],feature_number)))
  model.add(LSTM(layer_size2, activation = activation_function))
  model.add(Dense(dense_layer))
  model.compile(loss="mean_squared_error", optimizer = optimizer)

  return model

model = model_generation(50, 150, 1, X_train.shape[2], "tanh" )

def model_fit (epoch_size, batch_size, verbose_size):
  history = model.fit(X_train, y_train, epochs= epoch_size, batch_size= batch_size, validation_data=(X_valid, y_valid), verbose=verbose_size)
  return history

from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model_fit(150,16, 1)

test_data1 = pd.DataFrame(test_data)
train_data1 = pd.DataFrame(train_data)
past20 = train_data1.tail(20)

test_data_tailed = past20.append(test_data1, ignore_index = True)
test_data_tailed

inputs = scalar.fit_transform(test_data_tailed)
print(inputs.shape)

X_test = []
y_test = []

X_test_scaled = inputs[:,1:(inputs.shape[1])]
y_test_scaled = inputs[:,0:1]

X_test_scaled.shape, y_test_scaled.shape

for i in range(20, inputs.shape[0]):
    X_test.append(X_test_scaled[i-20:i])
    y_test.append(y_test_scaled[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape
#the desired shape

y_pred1 = model.predict(X_test)

y_pred1.shape , y_test.shape

y_preddf1 = pd.DataFrame(y_pred1)

num_new_cols = X_train.shape[2]
col_names = ['col{}'.format(i) for i in range(2, num_new_cols+2)]
y_preddf1[col_names] = 1

y_preddf1

y_preddf1 = (np.array(y_preddf1).reshape(-1,7))

y_preddf1 = scalar.inverse_transform(y_preddf1)

y_preddflast = pd.DataFrame(y_preddf1)
y_preddflast

predicted = y_preddflast[0]

predicted

y_testdf1 = pd.DataFrame(y_test)
y_testdf1

num_new_cols = X_train.shape[2]
col_names = ['col{}'.format(i) for i in range(2, num_new_cols+2)]
y_testdf1[col_names] = 1

y_testdf1

y_testdf1 = scalar.inverse_transform(y_testdf1)

y_testdflast = pd.DataFrame(y_testdf1)

ytest = y_testdflast[0]
ytest

mpe = np.mean((fixed_predicted - fixed_ytest) / fixed_ytest) * 100
average_actual_price = np.mean(fixed_ytest)
prmse = (np.sqrt(np.mean((fixed_predicted - fixed_ytest)**2)) / average_actual_price) * 100
mape = np.mean(np.abs((fixed_predicted - fixed_ytest) / fixed_ytest)) * 100
r2 = r2_score(fixed_ytest, fixed_predicted)

print("MPE: ", mpe)
print("PRMSE: ", prmse)
print("MAPE: ", mape)
print("R2: ", r2)

etf_std_dev = df1['Close'].pct_change().std()
vix_std_dev = df1['VIX'].pct_change().std()
volatility_ratio = etf_std_dev / vix_std_dev

print("Volatility Ratio for ETF:", volatility_ratio)
