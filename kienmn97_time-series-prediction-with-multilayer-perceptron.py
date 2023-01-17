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
plt.figure(figsize=(12, 8))

dataset.plot();
# Split the dataset into train and test set

train_size = int(dataset.shape[0] * 0.67)

train_df, test_df = dataset.iloc[:train_size, :], dataset.iloc[train_size:, :]
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

X_train, y_train = create_dataset(train_df.values, look_back=look_back)

X_test, y_test = create_dataset(test_df.values, look_back=look_back)
for i in range(5):

    print(X_train[i], y_train[i])
import tensorflow as tf

tf.__version__
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
model = Sequential()

model.add(Dense(8, input_dim=look_back, activation='relu'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=200, batch_size=2, verbose=2)
train_score = model.evaluate(X_train, y_train)

test_score = model.evaluate(X_test, y_test)

print('Train score: {} MSE'.format(train_score))

print('Test score: {} MSE'.format(test_score))
# Visualize the prediction

plt.figure(figsize=(12, 8))

train_prediction = model.predict(X_train)

train_stamp = np.arange(look_back, look_back + X_train.shape[0])

test_prediction = model.predict(X_test)

test_stamp = np.arange(2 * look_back + X_train.shape[0], len(dataset))

plt.plot(dataset, label='true values')

plt.plot(train_stamp, train_prediction, label='train prediction')

plt.plot(test_stamp, test_prediction, label = 'test_prediction')

plt.legend();
look_back = 3

X_train, y_train = create_dataset(train_df.values, look_back=look_back)

X_test, y_test = create_dataset(test_df.values, look_back=look_back)
model = Sequential()

model.add(Dense(12, input_dim=look_back, activation='relu'))

model.add(Dense(18, activation='relu'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=400, batch_size=2, verbose=2)
train_score = model.evaluate(X_train, y_train)

test_score = model.evaluate(X_test, y_test)

print('Train score: {} MSE'.format(train_score))

print('Test score: {} MSE'.format(test_score))
# Visualize the prediction

plt.figure(figsize=(12, 8))

train_prediction = model.predict(X_train)

train_stamp = np.arange(look_back, look_back + X_train.shape[0])

test_prediction = model.predict(X_test)

test_stamp = np.arange(2 * look_back + X_train.shape[0], len(dataset))

plt.plot(dataset, label='true values')

plt.plot(train_stamp, train_prediction, label='train prediction')

plt.plot(test_stamp, test_prediction, label = 'test_prediction')

plt.legend();