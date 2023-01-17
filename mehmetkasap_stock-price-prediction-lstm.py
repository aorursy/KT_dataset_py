# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Importing the libraries
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the training set
dataset_train = pd.read_csv('../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv')
dataset_train.head()
train = dataset_train.loc[:, ["Open"]].values
train
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)
train_scaled.shape
# train part
N = 1000
plt.plot(train_scaled[:N])
plt.show()
# we are going to predict the last 1226-N values 
plt.plot(train_scaled[N:])
plt.show()
# Creating a data structure with 50 timesteps and 1 output
X_train = []
y_train = []
timesteps = 50
for i in range(timesteps, N):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

# model
model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, 1), return_sequences=True)) # 128 lstm neuron(block)
model.add(LSTM(64, return_sequences=True)) # 64 lstm neuron(block)
model.add(Dropout(0.2))
model.add(LSTM(32)) # 128 lstm neuron(block)
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))
          
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=250, batch_size=32)
predicted_stock_price = []

#1
X_test = X_train[(N-51),:] # the last 
X_test = np.array(X_test)
X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
a = model.predict(X_test)
predicted_stock_price = np.append(predicted_stock_price, a)
X_test = np.append(X_test, predicted_stock_price) 
X_test
#2-7
for i in range(1,(1226-N)):
    X_test = X_test[1:51]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (1, 50, 1))
    a = model.predict(X_test)
    predicted_stock_price = np.append(predicted_stock_price, a)  
    X_test = np.append(X_test, predicted_stock_price[i])
predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1,1))
predicted_stock_price.shape
# train part
plt.plot(train[:N])
plt.show()
# we are going to predict 
plt.plot(train[N:], color = 'blue', label = 'Real Google Stock Price')
#plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
# Visualising the results
# plt.plot(train[1216:].reshape(-1,1), color = 'blue', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
