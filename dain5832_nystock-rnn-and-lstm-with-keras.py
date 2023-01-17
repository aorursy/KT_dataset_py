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
import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
price_adj_df = pd.read_csv("/kaggle/input/nyse/prices-split-adjusted.csv", index_col=0)
price_adj_df
price_adj_df.head()
price_adj_df.info()
price_adj_df.describe()
price_adj_df.head()
# visualize
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(price_adj_df[price_adj_df.symbol == "AAPL"].open.values, color='red', label='open')
plt.plot(price_adj_df[price_adj_df.symbol == "AAPL"].close.values, color='blue', label='close')
plt.plot(price_adj_df[price_adj_df.symbol == "AAPL"].low.values, color='yellow', label='low')
plt.plot(price_adj_df[price_adj_df.symbol == "AAPL"].high.values, color='green', label='high')
plt.title('stock price of Apple')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend()

#plt.subplot(1, 2, 2)
#plt.plot()
#plt.title('stock volume')
# function to create train, validation, test data given stck data and sequence length
def load_data(stock, seq_len):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])
        
    data = np.array(data)
    val_size = int(np.round(0.1*data.shape[0]))
    test_size = int(np.round(0.2*data.shape[0]))
    train_size = int(np.round(0.8*data.shape[0]))
    
    x_train = np.asarray(data[:train_size, :-1, :]).astype(np.float32)
    y_train = np.asarray(data[:train_size, -1, :]).astype(np.float32)
    
    x_valid = np.asarray(data[train_size:train_size+val_size, :-1, :]).astype(np.float32)
    y_valid = np.asarray(data[train_size:train_size+val_size, -1, :]).astype(np.float32)
    
    x_test = np.asarray(data[train_size+val_size:, :-1, :]).astype(np.float32)
    y_test = np.asarray(data[train_size+val_size:, -1, :]).astype(np.float32)
    
    return (x_train, y_train, x_valid, y_valid, x_test, y_test)
# choose one stock
aapl_stock = price_adj_df[price_adj_df.symbol == 'AAPL'].copy()
aapl_stock.drop(['symbol'], axis=1, inplace=True)
aapl_stock.drop(['volume'], axis=1, inplace=True)
cols = aapl_stock.columns.values
print(cols)
# normalize stock
aapl_stock_norm = aapl_stock.copy()
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
aapl_stock_norm['open'] = min_max_scaler.fit_transform(aapl_stock_norm.open.values.reshape(-1, 1))
aapl_stock_norm['close'] = min_max_scaler.fit_transform(aapl_stock_norm.close.values.reshape(-1, 1))
aapl_stock_norm['high'] = min_max_scaler.fit_transform(aapl_stock_norm.high.values.reshape(-1, 1))
aapl_stock_norm['low'] = min_max_scaler.fit_transform(aapl_stock_norm.low.values.reshape(-1, 1))

# create train, test data
seq_len = 15 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(aapl_stock_norm, seq_len)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)
plt.figure(figsize=(15, 5))
plt.plot(aapl_stock_norm.open.values, color='red', label='open')
plt.plot(aapl_stock_norm.close.values, color='blue', label='close')
plt.legend()
plt.title('aapl stock')
plt.ylabel('normalized price/volume')
plt.xlabel('time [days]')
# Basic Cell RNN in keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# build model
rnn_model = keras.Sequential()

rnn_model.add(layers.SimpleRNN(32, input_shape=(14, 4), return_sequences=True))
rnn_model.add(layers.SimpleRNN(32, return_sequences=False))
rnn_model.add(layers.Dense(4))
#rnn_model.add(layers.Activation('softmax'))

rnn_model.summary()
adam = optimizers.Adam(lr = 0.001)
rnn_model.compile(loss = 'mean_squared_error', optimizer=adam,
             metrics = ['accuracy'])
# train network
rnn_model.fit(x_train, y_train, batch_size = 32,
         epochs=20, validation_data=(x_valid, y_valid))
# predict and plot
rnn_preds = rnn_model.predict(x_test)
col_lst = list(aapl_stock_norm.columns)
figure, axes = plt.subplots(4,1,figsize=(10, 15))

for (i, col_name) in enumerate(col_lst):
    rnn_preds_col = min_max_scaler.inverse_transform(rnn_preds[:,i].reshape(-1,1))
    Ytest = min_max_scaler.inverse_transform(y_test[:,i].reshape(-1,1))

    testScore = math.sqrt(mean_squared_error(Ytest, rnn_preds_col))
    print(col_name, ' Test Score: %.2f RMSE' % (testScore))

    data = aapl_stock_norm[col_name].values.reshape(-1,1)
    data = np.reshape(data, (data.shape[0], 1))
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(aapl_stock_norm[col_name])-len(rnn_preds_col)-1:len(aapl_stock_norm['close'])-1] = rnn_preds_col
    
    axes[i].plot(min_max_scaler.inverse_transform(aapl_stock_norm[col_name].values.reshape(-1,1)))
    axes[i].plot(testPredictPlot)
    axes[i].set_title(col_name +' Prediction')
    plt.tight_layout()
    #plt.plot(rnn_preds)
# LSTM in keras
lstm_model = keras.Sequential()
lstm_model.add(layers.LSTM(200, input_shape=(14, 4), return_sequences=False))
#lstm_model.add(layers.LSTM(80))
lstm_model.add(layers.Dense(4))
lstm_model.summary()

adam = optimizers.Adam(lr = 0.001)
lstm_model.compile(loss='mean_squared_error', optimizer=adam,
                  metrics=["accuracy"])

lstm_model.fit(x_train, y_train, epochs=30,
              batch_size=32, verbose=1,
              validation_data=(x_valid, y_valid))
# predict and plot
lstm_preds = lstm_model.predict(x_test)

col_lst = list(aapl_stock_norm.columns)
figure, axes = plt.subplots(4,1,figsize=(10, 15))

for (i, col_name) in enumerate(col_lst):
    lstm_preds_col = min_max_scaler.inverse_transform(lstm_preds[:,i].reshape(-1,1))
    Ytest = min_max_scaler.inverse_transform(y_test[:,i].reshape(-1,1))

    testScore = math.sqrt(mean_squared_error(Ytest, lstm_preds_col))
    print(col_name, ' Test Score: %.2f RMSE' % (testScore))

    data = aapl_stock_norm[col_name].values.reshape(-1,1)
    data = np.reshape(data, (data.shape[0], 1))
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(aapl_stock_norm[col_name])-len(lstm_preds_col)-1:len(aapl_stock_norm['close'])-1] = lstm_preds_col
    
    axes[i].plot(min_max_scaler.inverse_transform(aapl_stock_norm[col_name].values.reshape(-1,1)))
    axes[i].plot(testPredictPlot)
    axes[i].set_title(col_name +' Prediction')
    plt.tight_layout()
    #plt.plot(rnn_preds)
# cross validation with lstm
lstm_model = keras.Sequential()
lstm_model.add(layers.LSTM(200, input_shape=(19, 4)))
lstm_model.add(layers.Dense(4))
lstm_model.summary()

adam = optimizers.Adam(lr = 0.001)
lstm_model.compile(loss='mean_squared_error', optimizer=adam,
                  metrics=["accuracy"])

lstm_model.fit(x_train, y_train, epochs=20,
              batch_size=32, verbose=1,
              validation_data=(x_valid, y_valid))
## Basic Cell RNN in tensorflow

index_in_epoch = 0
perm_array = np.arange(x_train.shape[0]) # 0부터 x_train길이까지
np.random.shuffle(perm_array)

# function to get the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch +=batch_size
    
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) # shuffle permutation array
        start = 0 # start next epoch
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

# parameters
n_steps = seq_len-1
n_inputs = 4
n_neurons = 200
n_outputs = 4
n_layers = 3
learning_rate = 0.001
batch_size = 64
n_epochs = 100
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

# use Basic RNN Cell
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                     activation=tf.nn.elu)
         for layer in range(n_layers)]

multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(n_categories + input_size + hidden_size,
                            hidden_size)
        self.i2o

# LSTM in pytorch
import torch
import torch.nn as nn

x_train = torch.FloatTensor(x_train).view(-1)