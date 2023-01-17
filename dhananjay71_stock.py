# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os as os



dataset_train = pd.read_csv("../input/Google_Stock_Price_Train.csv")

training_set = dataset_train.iloc[:,1:2].values



from sklearn.preprocessing import MinMaxScaler 



sc = MinMaxScaler(feature_range = (0,1))

training_set_scaled = sc.fit_transform(training_set)



X_train = []

y_train = []

for i in range(60, len(training_set_scaled)):

    X_train.append(training_set_scaled[i-60:i,0])

    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)



X_trian = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



dataset_train.head()
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



regressor = Sequential()



regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))



regressor.add(Dense(units = 1))



regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)



dataset_test = pd.read_csv("../input/Google_Stock_Price_Test.csv")

real_stock_price = dataset_test.iloc[:, 1:2].values



dataset_total = pd.concat((dataset_train['open'], dataset_test['open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1, 1)

X_test = []

for i in range(60, 80):

    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X-test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)



plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google stock price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()