# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

 

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

 

import keras

 

import pandas as pd

import numpy as np

import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import matplotlib.pyplot as plt

 

import matplotlib.pyplot as plt

from keras import models

from keras import layers

from keras.layers import Dense, Conv2D, Flatten

from keras.utils import to_categorical

from keras import backend as K

# %matplotlib inline

 

import os

 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

 

train_data = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

train_targets = train_data[['casual', 'registered', 'count']]

 

train_datetime_helper = train_data[['datetime']]

 

dt = pd.DatetimeIndex(train_data['datetime'])

# df.set_index(dt, inplace=True)

 

# train_data['date'] = dt.date

train_data['day'] = dt.day

train_data['month'] = dt.month

train_data['year'] = dt.year

train_data['hour'] = dt.hour

train_data['dow'] = dt.dayofweek

train_data['woy'] = dt.weekofyear

 

 

print(train_data)

 

train_data = train_data.drop(['casual', 'registered', 'count', 'datetime'], axis=1)

 

# print(train_data)

 

test_data = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

 

test_datetime_helper = test_data[['datetime']]

 

dt = pd.DatetimeIndex(test_data['datetime'])

 

test_data['day'] = dt.day

test_data['month'] = dt.month

test_data['year'] = dt.year

test_data['hour'] = dt.hour

test_data['dow'] = dt.dayofweek

test_data['woy'] = dt.weekofyear

 

test_data = test_data.drop(['datetime'], axis=1)

 

mean = train_data.mean(axis=0)

train_data -= mean

std = train_data.std(axis=0)

train_data /= std

 

test_data -= mean

test_data /= std

 

print(mean)

print(std)

 

 

def build_model():

    model = models.Sequential()

 

    model.add(layers.Dense(6493, activation='relu', input_shape=(train_data.shape[1],))) 

 

    model.add(layers.Dropout(0.34))

 

    model.add(layers.Dense(100, activation='relu'))

    # model.add(layers.Dense(95, activation='relu'))

 

    model.add(layers.Dropout(0.28))

 

    model.add(layers.Dense(32, activation='relu'))

 

    model.add(layers.Dropout(0.10))

 

    model.add(layers.Dense(3))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

 

    return model

 

 

test_targets = pd.DataFrame()

test_targets['casual'] = test_data['day']

test_targets['casual'] = 0

test_targets['registered'] = test_data['day']

test_targets['registered'] = 0

test_targets['count'] = test_data['day']

test_targets['count'] = 0

print(test_targets)

 

now = datetime.datetime.now()  # fixme

print(now)

 

model = build_model()

model.fit(train_data, train_targets,

          epochs=50, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)



print(test_mae_score)

 

my_predict = model.predict(test_data)

answer = pd.DataFrame(data=my_predict, index=np.array(range(1, 6494)), columns=np.array(range(1, 4)))

 

answer['count'] = answer[3]

 

 

my_answer = pd.DataFrame()

 

datetime_helper = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

my_answer['datetime'] = datetime_helper['datetime']

my_answer['count'] = answer['count']

 

my_answer.loc[my_answer['count'] < 0, 'count'] = my_answer['count'] * (-1)

my_answer.loc[my_answer['datetime'] == '2011-01-20 00:00:00', 'count'] = 1

my_answer.loc[my_answer['count'] < 1, 'count'] = 1

 

print(my_answer)

print('output1')

my_answer.to_csv('output1.csv', index=False)

print('output2')

 

now1 = datetime.datetime.now()

print(now1)