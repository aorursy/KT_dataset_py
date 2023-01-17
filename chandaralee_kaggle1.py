import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, Dropout, GRU, Bidirectional

from keras.optimizers import SGD

import math

from sklearn.metrics import mean_squared_error

print(os.listdir("../input"))

import keras


dataset = pd.read_csv('../input/historical_stock_prices.csv', index_col='date', parse_dates=['date'])
dataset.head()
dataset = [1,2,3,4,5,6,7,8,9]
dataset
dataset1= np.array(dataset)
dataset1

dataset2 = []
dataset2 = dataset1.reshape((-1),1)

dataset2
d = [1,2]

d = np.array(d)

d
d1=d.reshape((-1),1)

d1
sc = MinMaxScaler(feature_range=(0,1))


test = sc.fit_transform(d1)

test