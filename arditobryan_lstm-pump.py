# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def labelencoder(df, partitions):
  from sklearn.preprocessing import LabelEncoder
  for col in partitions:
    print(col)
    le = LabelEncoder()
    k = df.pop(col)
    le.fit(k)
    k = pd.DataFrame(le.transform(k))
    df = pd.concat([df, k] , axis=1)
  return df
import pandas as pd
X = pd.read_csv('../input/lstm-sensor/sensor.csv')

X =  X.drop(['Unnamed: 0', 'sensor_50', 'sensor_15'], axis=1)
X = labelencoder(X, ['machine_status'])
X
#without dropna
from datetime import *
X['timestamp'] = X['timestamp'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
X['timestamp'][0]

import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x=X['timestamp'], y=X[0])
def labelencoder(df, partitions):
  from sklearn.preprocessing import LabelEncoder
  for col in partitions:
    print(col)
    le = LabelEncoder()
    k = df.pop(col)
    le.fit(k)
    k = pd.DataFrame(le.transform(k))
    df = pd.concat([df, k] , axis=1)
  return df
#preprocessing
import pandas as pd
X = pd.read_csv('../input/lstm-sensor/sensor.csv')

X =  X.drop(['Unnamed: 0', 'timestamp', 'sensor_50', 'sensor_15'], axis=1)
X = labelencoder(X, ['machine_status'])

#filling Nan
X.interpolate(method='linear', inplace=True)
X

#making backup
K = X.copy()
X
def scale(partitions, scaler='MinMaxScaler', df=pd.DataFrame(), to_float=False, return_df=False):
  #return_df == True, allora output = scaler, df
  #return_df == False, allora output = df
  from sklearn.preprocessing import RobustScaler
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.preprocessing import StandardScaler
  
  if scaler == 'RobustScaler':
    f_transformer = RobustScaler()
  elif scaler == 'MinMaxScaler':
    f_transformer = MinMaxScaler(feature_range=(0, 1))
  elif scaler == 'StandardScaler':
    f_transformer = StandardScaler()
  
  #partitions = 'all_df', le fa tutte insieme e trasforma il df in un numpy.array
  if partitions == 'all_df':
    if to_float == True:
      df = df.astype('float32')
    if df.empty == True:
      X = df.copy()
    #tutto df deve essere con float32
    df_col = df.columns
    df = f_transformer.fit_transform(df.values) #ne esce un inspiegabile numpy array
    df = pd.DataFrame(df)
    df.columns = df_col
    if return_df == True:
      return f_transformer, df
    else:
      X = df.copy()
    return f_transformer
  else:
    #partitions = ['col1', 'col2'], fa solo partizioni specificate
    pass

def transform_to_stationary(df):
  #create a differenced series
  def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
      value = dataset[i] - dataset[i - interval]
      diff.append(value)
    return pd.DataFrame(diff)
  
  df = df.values #al di fuori delle funzioni voglio operare solo su un DataFrame
  df = difference(df, 1) #X ritorna ad essere un df
  return df

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, drop_col=False, y_var=1):
  n_features = int(len(data.columns))
  n_vars = 1 if type(data) is list else data.shape[1]
  df = pd.DataFrame(data)
  cols, names = list(), list()
  # input sequence (t-n, ... t-1)
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
  # forecast sequence (t, t+1, ... t+n)
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
    data = agg.copy()
    
  if drop_col == True:
    tot = n_features*n_in+n_features #24+8 = 32

    y_name = list(data.columns)[n_features*n_in-1 + y_var]
    y = data[y_name]
    for i in range(n_features*n_in, tot):
      data.drop(data.columns[[tot-n_features]], axis=1, inplace=True)
    data = pd.concat([data, y], axis=1)
  return data

def split(df, test_size):
  df = df.values
  len_df = df.shape[0]
  test_size = int(len_df*test_size)
  train, test = df[0:-test_size], df[-test_size:]
  return train, test
#retrieve backup
X = K.copy()

#   scaling, no stationary
raw_values = X.copy().values
scaler, X = scale('all_df', scaler='MinMaxScaler', df=X, to_float=True, return_df=True)
#X = transform_to_stationary(X)
X = series_to_supervised(X, 10, 1, drop_col=False)

#X, y
y = pd.DataFrame(X.pop('var51(t)'))
var_list = ['var'+str(x)+'(t)' for x in range(1, 51)]
X = X.drop(var_list, axis=1)

#train, test
X_train_, X_test_ = split(X, .1)
y_train_, y_test_ = split(y, .1) #sembra non servire a nulla
print(X_train_.shape, X_test_.shape, y_train_.shape, y_test_.shape)
X
#retrieve backup
X = K.copy()

#   scaling, stationary
raw_values = X.copy().values
scaler, X = scale('all_df', scaler='MinMaxScaler', df=X, to_float=True, return_df=True)
X = transform_to_stationary(X)
X = series_to_supervised(X, 10, 1, drop_col=False)

#X, y
y = pd.DataFrame(X.pop('var51(t)'))
var_list = ['var'+str(x)+'(t)' for x in range(1, 51)]
X = X.drop(var_list, axis=1)

#train, test
X_train, X_test = split(X, .1)
y_train, y_test = split(y, .1) #sembra non servire a nulla
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X
#reshape [samples, n_input_timesteps, n_features]
X_train = X_train.reshape((198279, 510, 1)) #che e.X_ e e.X abbiano eguali righe o una in più è irrilevante, non ci sconvolgiamo per questo
y_train = y_train.reshape((198279, 1, 1))
print(X_train.shape, y_train.shape)
#ogni singolo sample ha dimensioni [1, 6, 1]

#LSTM
%tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', batch_input_shape=(1, 510, 1)))
model.add(RepeatVector(1)) #numero di output
model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mse', optimizer='adam')
#model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=3, batch_size=1, verbose=2, shuffle=False)
model.reset_states()

X_test = X_test.reshape(22030, 510, 1)
y_test = y_test.reshape(22030, 1, 1)
print(X_test.shape, y_test.shape)