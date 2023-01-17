import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import sklearn

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
data = pd.read_csv('../input/new-tata-dataset/NSE-TATAGLOBAL.csv')
training_set = data.iloc[:, 1: 2].values
print(training_set)
data.head(5)
training_set
sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)
x_train = []

y_train = []

for i in range(60, 2035):

    x_train.append(training_set_scaled[i - 60: i, 0])

    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)



x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))



regressor.add(Dense(units = 1))



regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)
data_test = pd.read_csv('../input/sample-stock-test/Sample_Stock_test.csv')

real_stock_price = data_test.iloc[:, 1:2].values
data_total = pd.concat((data['Open'], data_test['Open']), axis = 0)

inputs = data_total[len(data_total) - len(data_test) - 60:].values

inputs = inputs.reshape(-1, 1)

inputs = sc.transform(inputs)
x_test = []

for i in range(60, 76):

    x_test.append(inputs[i - 60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plt.plot(real_stock_price, color = 'black', label = 'Tata Stock Price')

plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Tata Stock Price')

plt.title('Tata Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel = ('Tata Stock Price')

plt.legend()

plt.show()