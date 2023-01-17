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
from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import sqrt

from numpy import split

from numpy import array

from pandas import read_csv

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM
data = pd.read_csv("/kaggle/input/rk-puram-ambient-air/trainset.csv", header=0)

data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')

data = data.set_index('datetime')

data['Day'] = data.index.day

cols = data.columns.tolist()

cols.insert(0, cols.pop(cols.index('Weekday')))

cols.insert(1, cols.pop(cols.index('Day')))

cols.insert(2, cols.pop(cols.index('Month')))

cols.insert(3, cols.pop(cols.index('Year')))

cols.insert(4, cols.pop(cols.index('Hour')))

data = data.reindex(columns= cols)

data = data.drop('Weekday', axis=1)
data = data.values

n_train_hours = 365 * 24

train = data[:n_train_hours, :]

test = data[n_train_hours:, :]
def to_supervised(data, n_input=24, n_out=24):

    X, y = list(), list()

    in_start = 0

    # step over the entire history one time step at a time

    for _ in range(len(data)):

        # define the end of the input sequence

        in_end = in_start + n_input

        out_end = in_end + n_out

        # ensure we have enough data for this instance

        if out_end < len(data):

            X.append(data[in_start:in_end, :])

            y.append(data[in_end:out_end, 17])

                # move along one time step

        in_start += 1

    return array(X), array(y)
train_x ,train_y  = to_supervised(train)

test_x ,test_y  = to_supervised(train)
verbose, epochs, batch_size, n_output = 1, 800, 72 , 24

n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

# define model

model = Sequential()

model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))

model.add(Dense(100, activation='relu'))

model.add(Dense(n_outputs))

model.compile(loss='mae', optimizer='adam')

# fit network

history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_x, test_y), verbose=verbose)
from matplotlib import pyplot as plt

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()