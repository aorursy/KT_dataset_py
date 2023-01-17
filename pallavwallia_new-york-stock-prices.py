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
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt

df_prices=pd.read_csv('../input/nyse/prices.csv', header=0)
df_prices_split=pd.read_csv('../input/nyse/prices-split-adjusted.csv', header=0)
df_prices_split
df_prices.info()
df_prices.describe()
df_prices.head()
df_prices.isnull().sum()
df_prices['symbol'].unique()
df_prices['symbol'].value_counts()
plt.figure(figsize=(15, 5));
plt.plot(df_prices[df_prices.symbol == 'AAPL'].open.values, color='green', label='open')
plt.plot(df_prices[df_prices.symbol == 'AAPL'].close.values, color='red', label='close')
plt.plot(df_prices[df_prices.symbol == 'AAPL'].low.values, color='yellow', label='low')
plt.plot(df_prices[df_prices.symbol == 'AAPL'].high.values, color='blue', label='high')
plt.plot(df_prices[df_prices.symbol == 'CLX'].open.values, color='green', label='open')
plt.plot(df_prices[df_prices.symbol == 'CLX'].close.values, color='red', label='close')
plt.plot(df_prices[df_prices.symbol == 'CLX'].low.values, color='yellow', label='low')
plt.plot(df_prices[df_prices.symbol == 'CLX'].high.values, color='blue', label='high')
plt.plot(df_prices[df_prices.symbol == 'ETR'].open.values, color='green', label='open')
plt.plot(df_prices[df_prices.symbol == 'ETR'].close.values, color='red', label='close')
plt.plot(df_prices[df_prices.symbol == 'ETR'].low.values, color='yellow', label='low')
plt.plot(df_prices[df_prices.symbol == 'ETR'].high.values, color='blue', label='high')
apple = df_prices[df_prices['symbol']=='AAPL']
apple_stock_prices = apple.close.values.astype('float32')
apple_stock_prices = apple_stock_prices.reshape(1762, 1)
apple_stock_prices.shape

df_prices['symbol'][df_prices['symbol']=='AAPL'].value_counts()
scaler = MinMaxScaler(feature_range=(0, 1))
apple_stock_prices = scaler.fit_transform(apple_stock_prices)
apple_stock_prices
train_size = int(len(apple_stock_prices) * 0.90)
test_size = len(apple_stock_prices) - train_size
train = apple_stock_prices[0:train_size,:] 
test=apple_stock_prices[train_size:len(yahoo_stock_prices),:]
train.shape
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()
d=0.8
model.add(LSTM(128, input_shape=1))#, return_sequences=True))
model.add(Dropout(d))
        
model.add(LSTM(128, input_shape=1, return_sequences=False))
model.add(Dropout(d))
        
model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    
model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
