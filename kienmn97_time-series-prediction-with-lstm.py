# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv('/kaggle/input/international-airline-passengers/international-airline-passengers.csv',

                      usecols=[1], engine='python', skipfooter=3)

dataset.columns = ['Passengers']
dataset.head()
dataset.plot(figsize=(12, 8));
# Split the dataset into train and test set

dataset = dataset.values

train_size = int(dataset.shape[0] * 0.67)

train_df, test_df = dataset[:train_size, :], dataset[train_size:, :]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

train_df_rescaled = scaler.fit_transform(train_df)

test_df_rescaled = scaler.transform(test_df)
# Convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):

    m = len(dataset)

    X = []

    y = []

    for i in range(look_back, m):

        X.append(dataset[i - look_back: i, 0])

        y.append(dataset[i, 0])

    return np.array(X), np.array(y)
look_back = 1

X_train_rescaled, y_train_rescaled = create_dataset(train_df_rescaled, look_back=look_back)

X_test_rescaled, y_test_rescaled = create_dataset(test_df_rescaled, look_back=look_back)
# Reshape array to have the form of [samples, time_steps, features]

X_train_rescaled = np.reshape(X_train_rescaled, (X_train_rescaled.shape[0], X_train_rescaled.shape[1], 1))

X_test_rescaled = np.reshape(X_test_rescaled, (X_test_rescaled.shape[0], X_test_rescaled.shape[1], 1))
import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import LSTM, Dense
model = Sequential()

model.add(LSTM(4, input_shape=(look_back, 1)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train_rescaled, y_train_rescaled, epochs=100, batch_size=1, verbose=2)
from sklearn.metrics import mean_squared_error

train_prediction = model.predict(X_train_rescaled)

train_prediction = scaler.inverse_transform(train_prediction)

y_train = scaler.inverse_transform(y_train_rescaled.reshape(-1, 1))

test_prediction = model.predict(X_test_rescaled)

test_prediction = scaler.inverse_transform(test_prediction)

y_test = scaler.inverse_transform(y_test_rescaled.reshape(-1, 1))

train_score = mean_squared_error(y_train, train_prediction)

test_score = mean_squared_error(y_test, test_prediction)

print('Train score: {} MSE'.format(train_score))

print('Test score: {} MSE'.format(test_score))
# Visualize the prediction

plt.figure(figsize=(12, 8))

train_stamp = np.arange(look_back, look_back + X_train_rescaled.shape[0])

test_stamp = np.arange(2 * look_back + X_train_rescaled.shape[0], len(dataset))

plt.plot(dataset, label='true values')

plt.plot(train_stamp, train_prediction, label='train prediction')

plt.plot(test_stamp, test_prediction, label = 'test_prediction')

plt.ylabel('Passengers')

plt.xlabel('Time stamp')

plt.legend();
look_back = 3

X_train_rescaled, y_train_rescaled = create_dataset(train_df_rescaled, look_back=look_back)

X_test_rescaled, y_test_rescaled = create_dataset(test_df_rescaled, look_back=look_back)
# Reshape array to have the form of [samples, time_steps, features]

X_train_rescaled = np.reshape(X_train_rescaled, (X_train_rescaled.shape[0], X_train_rescaled.shape[1], 1))

X_test_rescaled = np.reshape(X_test_rescaled, (X_test_rescaled.shape[0], X_test_rescaled.shape[1], 1))
model = Sequential()

model.add(LSTM(4, input_shape=(look_back, 1)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train_rescaled, y_train_rescaled, epochs=100, batch_size=1, verbose=2)
from sklearn.metrics import mean_squared_error

train_prediction = model.predict(X_train_rescaled)

train_prediction = scaler.inverse_transform(train_prediction)

y_train = scaler.inverse_transform(y_train_rescaled.reshape(-1, 1))

test_prediction = model.predict(X_test_rescaled)

test_prediction = scaler.inverse_transform(test_prediction)

y_test = scaler.inverse_transform(y_test_rescaled.reshape(-1, 1))

train_score = mean_squared_error(y_train, train_prediction)

test_score = mean_squared_error(y_test, test_prediction)

print('Train score: {} MSE'.format(train_score))

print('Test score: {} MSE'.format(test_score))
# Visualize the prediction

plt.figure(figsize=(12, 8))

train_stamp = np.arange(look_back, look_back + X_train_rescaled.shape[0])

test_stamp = np.arange(2 * look_back + X_train_rescaled.shape[0], len(dataset))

plt.plot(dataset, label='true values')

plt.plot(train_stamp, train_prediction, label='train prediction')

plt.plot(test_stamp, test_prediction, label = 'test_prediction')

plt.ylabel('Passengers')

plt.xlabel('Time stamp')

plt.legend();
epochs = 100

batch_size = 1



model = Sequential()

model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')



for i in range(epochs):

    model.fit(X_train_rescaled, y_train_rescaled, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)

    model.reset_states()
train_prediction = model.predict(X_train_rescaled, batch_size=batch_size)

train_prediction = scaler.inverse_transform(train_prediction)

y_train = scaler.inverse_transform(y_train_rescaled.reshape(-1, 1))

test_prediction = model.predict(X_test_rescaled, batch_size=batch_size)

test_prediction = scaler.inverse_transform(test_prediction)

y_test = scaler.inverse_transform(y_test_rescaled.reshape(-1, 1))

train_score = mean_squared_error(y_train, train_prediction)

test_score = mean_squared_error(y_test, test_prediction)

print('Train score: {} MSE'.format(train_score))

print('Test score: {} MSE'.format(test_score))
# Visualize the prediction

plt.figure(figsize=(12, 8))

train_stamp = np.arange(look_back, look_back + X_train_rescaled.shape[0])

test_stamp = np.arange(2 * look_back + X_train_rescaled.shape[0], len(dataset))

plt.plot(dataset, label='true values')

plt.plot(train_stamp, train_prediction, label='train prediction')

plt.plot(test_stamp, test_prediction, label = 'test_prediction')

plt.ylabel('Passengers')

plt.xlabel('Time stamp')

plt.legend();
epochs = 100

batch_size = 1



model = Sequential()

model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))

model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')



for i in range(epochs):

    model.fit(X_train_rescaled, y_train_rescaled, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)

    model.reset_states()
train_prediction = model.predict(X_train_rescaled, batch_size=batch_size)

train_prediction = scaler.inverse_transform(train_prediction)

y_train = scaler.inverse_transform(y_train_rescaled.reshape(-1, 1))

test_prediction = model.predict(X_test_rescaled, batch_size=batch_size)

test_prediction = scaler.inverse_transform(test_prediction)

y_test = scaler.inverse_transform(y_test_rescaled.reshape(-1, 1))

train_score = mean_squared_error(y_train, train_prediction)

test_score = mean_squared_error(y_test, test_prediction)

print('Train score: {} MSE'.format(train_score))

print('Test score: {} MSE'.format(test_score))
# Visualize the prediction

plt.figure(figsize=(12, 8))

train_stamp = np.arange(look_back, look_back + X_train_rescaled.shape[0])

test_stamp = np.arange(2 * look_back + X_train_rescaled.shape[0], len(dataset))

plt.plot(dataset, label='true values')

plt.plot(train_stamp, train_prediction, label='train prediction')

plt.plot(test_stamp, test_prediction, label = 'test_prediction')

plt.ylabel('Passengers')

plt.xlabel('Time stamp')

plt.legend();