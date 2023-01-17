import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
# import train data
data_train = pd.read_csv('../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv')
data_train.head()
train = data_train.loc[:, ['Open']].values
print(train)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)
print(train_scaled)
# data plotting
plt.plot(train_scaled)
plt.show()
train_scaled.shape
# the train data split x and y
X_train = []
Y_train = []
timesteps = 50
for i in range(timesteps, 1226):
    X_train.append(train_scaled[i - timesteps:i, 0])
    Y_train.append(train_scaled[i, 0])
x_train, y_train = np.array(X_train), np.array(Y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)
print(y_train.shape)
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
# create RNN model

model_rnn = Sequential()

model_rnn.add(SimpleRNN(units = 50, activation = 'tanh', return_sequences = True, input_shape = (x_train.shape[1], 1)))
model_rnn.add(Dropout(0.25))

model_rnn.add(SimpleRNN(units = 50, activation = 'tanh', return_sequences = True))
model_rnn.add(Dropout(0.25))

model_rnn.add(SimpleRNN(units = 50, activation = 'tanh', return_sequences = True))
model_rnn.add(Dropout(0.25))

model_rnn.add(SimpleRNN(units = 50, activation = 'tanh', return_sequences = True))
model_rnn.add(Dropout(0.25))

model_rnn.add(SimpleRNN(units = 50))
model_rnn.add(Dropout(0.25))

model_rnn.add(Dense(units = 1))

model_rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
model_rnn.fit(x_train, y_train, epochs = 100, batch_size = 32)
# import test data
data_test = pd.read_csv('../input/Uniqlo(FastRetailing) 2017 Test - stocks2017.csv')
data_test.head(10)
test_price = data_test.loc[:, ['Open']].values
print(test_price)
data_total = pd.concat((data_train['Open'], data_test['Open']), axis = 0)
inputs = data_total[len(data_total) - len(data_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)
print(inputs.shape)
# rnn model predict
X_test = []
for i in range(timesteps, 57):
    X_test.append(inputs[i - timesteps:i, 0])

x_test = np.array(X_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted = model_rnn.predict(x_test)
predicted = scaler.inverse_transform(predicted)
# real data end rnn model predict plotting
plt.plot(test_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
data_train = pd.read_csv('../input/Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv')
data_train.head()
train = data_train.loc[:, ['Open']].values
print(train.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)
print(train_scaled)
X_train = []
y_train = []
timesteps = 50
for i in range(timesteps, 1226):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
# import keras library
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
model_lstm = Sequential()
model_lstm.add(LSTM(150,return_sequences=True,input_shape=(timesteps, 1) ))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(100, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(80))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer = 'adam')
model_lstm.fit(X_train, y_train, epochs = 250, batch_size = 32)
# lstm model predict
predicted = model_lstm.predict(x_test)
predicted = scaler.inverse_transform(predicted)
# real data and lstm model predict polotting
plt.plot(test_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()