import pandas as pd

import numpy as np


data = pd.read_csv('./../input/sav_2013_2017.csv')

data.head()
traffic = data[['av_factor', 'hour', 'weekday', 'hits']][-10000:]
traffic.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

hit_scaler = MinMaxScaler(feature_range=(0,1))



length = len(traffic)



traffic.values[:,0] = scaler.fit_transform(np.matrix(traffic.av_factor).T).reshape((length))

traffic.values[:,1] = scaler.fit_transform(np.matrix(traffic.hour).T).reshape((length))

traffic.values[:,2] = scaler.fit_transform(np.matrix(traffic.weekday).T).reshape((length))



traffic.values[:,3] = hit_scaler.fit_transform(np.matrix(traffic.hits).T).reshape((length))
traffic.hits.mean(), traffic.hits.std()
window_size = 2
def get_rolling_window(data, window_size):

    x, y, val = [], [], len(data) - window_size

    for z in range(val-1):

        x.append(data.values[z:(z + window_size)])

        y.append(data.values[z + window_size,-1:])

    return np.array(x), np.array(y)
split_factor = 0.8

split_row = int(len(traffic) * split_factor)

split_row
train_data, test_data = traffic[:split_row], traffic[split_row:]

X_train, y_train = get_rolling_window(train_data, window_size)

X_test, y_test = get_rolling_window(test_data, window_size)



len(X_train), len(X_test)
from keras.models import Sequential

from keras.layers import Dense, LSTM, SimpleRNN
model = Sequential()

batch_size = 1

model.add(SimpleRNN(4, batch_input_shape=(batch_size, window_size, 4), stateful=True, return_sequences=True))

model.add(LSTM(3, batch_input_shape=(batch_size, window_size, 4), stateful=True, return_sequences=True))

model.add(LSTM(5,batch_input_shape=(batch_size, window_size, 4), stateful=True))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=batch_size, shuffle=False)
def inverse_transform(pred):    

    return hit_scaler.inverse_transform(pred)
pred_x_train = model.predict(X_train, batch_size)

pred_train = inverse_transform(pred_x_train)

pred_train[pred_train < 0] = 0.
y_train = np.float_(y_train)

y_train_inv = inverse_transform(y_train)
pred_x_test = model.predict(X_test, batch_size)

pred_test = inverse_transform(pred_x_test)

pred_test[pred_test < 0] = 0.
y_test = np.float_(y_test)

y_test_inv = inverse_transform(y_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error

import math
def get_mse(real, pred):

    return math.sqrt(mean_squared_error(real, pred))



def get_mae(real, pred):

    return mean_absolute_error(real, pred)
train_mse = get_mse(y_train_inv, pred_train)

test_mse = get_mse(y_test_inv, pred_test)



print('Train Score: %.2f RMSE' % (train_mse))

print('Test Score: %.2f RMSE' % (test_mse))
train_mae = get_mae(y_train_inv, pred_train)

test_mae = get_mae(y_test_inv, pred_test)



print('Train Score: %.2f MAE' % (train_mae))

print('Test Score: %.2f MAE' % (test_mae))
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
line_test_pred = np.reshape(pred_test, pred_test.shape[0])

line_test_real = np.reshape(y_test_inv, y_test_inv.shape[0])
plt.figure(figsize=(20,10))

plt.plot(line_test_real[:300], color='blue',label='Original')

plt.plot(line_test_pred[:300], color='red',label='Prediction')

plt.legend(loc='best')

plt.title('Test - Comparison')

plt.show()
line_train_pred = np.reshape(pred_train, pred_train.shape[0])

line_train_real = np.reshape(y_train_inv, y_train_inv.shape[0])
plt.figure(figsize=(20,10))

plt.plot(line_train_real[-300:], color='blue',label='Original')

plt.plot(line_train_pred[-300:], color='red',label='Prediction')

plt.legend(loc='best')

plt.title('Train - Comparison')

plt.show()