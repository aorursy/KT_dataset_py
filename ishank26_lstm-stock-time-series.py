# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import math

import sklearn

import sklearn.preprocessing

import datetime

import os

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import LSTM,Dense

from sklearn.preprocessing import MinMaxScaler

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf



# split data in 60%/20%/20% train/validation/test sets

valid_set_size_percentage = 20 

test_set_size_percentage = 20 



#display parent directory and working directory

print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())));

print(os.getcwd()+':', os.listdir(os.getcwd()));
# import all stock prices 

df = pd.read_csv("../input/prices-split-adjusted.csv", index_col = 0)

df.info()

df.head()



# number of different stocks

print('\n Number of stocks: ', len(list(set(df.symbol))))

print('Some stock symbols: ', list(set(df.symbol))[:10])
# self check data

df.tail(4)
start = min(df.index.tolist())

end = max(df.index.tolist())

print("Start date: ", start, ", End date: ", end)
# Get data summary

df.describe()
# Get data info.Useful to see datatypes of features

df.info()
ticker = 'MSFT'
# Visualize data

plt.figure(figsize=(20, 8));

plt.subplot(1,2,1);

plt.plot(df[df.symbol == ticker].open.values, color='red', label='open')

plt.plot(df[df.symbol == ticker].close.values, color='green', label='close')

plt.plot(df[df.symbol == ticker].low.values, color='blue', label='low')

plt.plot(df[df.symbol == ticker].high.values, color='black', label='high')

plt.title('Stock price')

plt.xlabel('time [days]')

plt.ylabel('price')

plt.legend(loc='best')

#plt.show()



plt.subplot(1,2,2);

plt.plot(df[df.symbol == ticker].volume.values, color='black', label='volume')

plt.title('stock volume')

plt.xlabel('time [days]')

plt.ylabel('volume')

plt.legend(loc='best');
subset = df[df.symbol == ticker].loc[:,list(df.keys())[1:]]
# Significant coorelation between lags 

acf = plot_acf(subset['open'], lags = 20)
# No significant pattern in pacf

pacf = plot_pacf(subset['open'], lags = 20)
w = 20 # window size 
roll_df = subset.rolling(window=w).mean()

roll_df['symbol'] = df[df.symbol == ticker].loc[:,'symbol']

roll_df.dropna(inplace=True)

roll_df.describe()
plt.figure(figsize=(20, 10));

plt.subplot(1,2,1);

plt.plot(roll_df[roll_df.symbol == ticker].open.values, color='red', label='open')

plt.plot(roll_df[roll_df.symbol == ticker].close.values, color='green', label='close')

plt.plot(roll_df[roll_df.symbol == ticker].low.values, color='blue', label='low')

plt.plot(roll_df[roll_df.symbol == ticker].high.values, color='black', label='high')

plt.title('Stock price for {}-day rolling avg.'.format(w))

plt.xlabel('time [days]')

plt.ylabel('price')

plt.legend(loc='best')

#plt.show()



plt.subplot(1,2,2);

plt.plot(roll_df[roll_df.symbol == ticker].volume.values, color='black', label='volume')

plt.title('Stock volume for {}-day rolling avg.'.format(w))

plt.xlabel('time [days]')

plt.ylabel('volume')

plt.legend(loc='best');
# function for min-max normalization of stock

min_max_scaler = sklearn.preprocessing.MinMaxScaler()

def normalize_data(df):

    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))

    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))

    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))

    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))

    return df
# choose one stock

df_stock = df[df.symbol == ticker].copy()

df_stock.drop(['symbol'],1,inplace=True)

df_stock.drop(['volume'],1,inplace=True)



cols = list(df_stock.columns.values)

print('df_stock.columns.values = ', cols)



# normalize stock

df_stock_norm = df_stock.copy()

df_stock_norm = normalize_data(df_stock_norm)

df_stock_norm.describe()
# function to create train, validation, test data given stock data and sequence length

def load_data(stock, seq_len):

    data_raw = stock.as_matrix() # convert to numpy array

    data = []

    

    # create all possible sequences of length seq_len

    for index in range(len(data_raw) - seq_len): 

        data.append(data_raw[index: index + seq_len])

    

    data = np.array(data);

    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));  

    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));

    train_set_size = data.shape[0] - (valid_set_size + test_set_size);

    

    x_train = data[:train_set_size,:-1,:]

    y_train = data[:train_set_size,-1,:]

    

    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]

    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]

    

    x_test = data[train_set_size+valid_set_size:,:-1,:]

    y_test = data[train_set_size+valid_set_size:,-1,:]

    

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

# create train, test data

seq_len = 19 # choose sequence length

x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len+1)

print('x_train.shape = ',x_train.shape)

print('y_train.shape = ', y_train.shape)

print('x_valid.shape = ',x_valid.shape)

print('y_valid.shape = ', y_valid.shape)

print('x_test.shape = ', x_test.shape)

print('y_test.shape = ',y_test.shape)
# Choose only open prices

x_train, y_train, x_valid, y_valid, x_test, y_test = x_train[:,:,0], y_train[:,0], x_valid[:,:,0], y_valid[:,0], x_test[:,:,0], y_test[:,0]

print('x_train.shape = ',x_train.shape)

print('y_train.shape = ', y_train.shape)

print('x_valid.shape = ',x_valid.shape)

print('y_valid.shape = ', y_valid.shape)

print('x_test.shape = ', x_test.shape)

print('y_test.shape = ',y_test.shape)
#Build the model

# Predict every 20th time step

model = Sequential()

model.add(LSTM(256,input_shape=(seq_len,1)))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
#Reshape data for (Sample,Timestep,Features) 

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))

x_valid = x_valid.reshape((x_valid.shape[0],x_valid.shape[1],1))

x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))
epoch_size = 100
#Fit model with history to check for overfitting

history = model.fit(x_train,y_train,epochs=epoch_size,validation_data=(x_valid,y_valid),shuffle=False)
# Visualise predictions

Xt = model.predict(x_test)

plt.figure(figsize=(15, 5));

plt.plot( np.arange(y_test.shape[0]), min_max_scaler.inverse_transform(y_test.reshape(-1,1)),label='Original')

plt.plot(np.arange(y_test.shape[0]), min_max_scaler.inverse_transform(Xt), label ='Predictions')

plt.xlabel('Days')

plt.ylabel('Price')

plt.legend(loc='best')

plt.show()
scores = model.evaluate(x_test, y_test, verbose = 0)

print("Test accuracy: ", scores[1])
# Visualize loss

plt.figure(figsize=(15, 5));

plt.plot(history.history['loss'], label='Training loss')

plt.plot(history.history['val_loss'], label ='Validation loss')

plt.xlabel('epochs')

plt.ylabel('Loss')

plt.legend(loc='best')

plt.show()