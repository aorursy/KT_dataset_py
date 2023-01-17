import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, Dropout, GRU, Bidirectional, LSTM

from keras.optimizers import SGD

import math

from sklearn.metrics import mean_squared_error

import keras
# Load dataset from csv file

dataset = pd.read_csv('../input/historical_stock_prices.csv', index_col='date', parse_dates=['date'])
# Separate train/test

trainSet = dataset['2015':'2017'].sort_values(by=['ticker','date'])

testSet = dataset['2018':].sort_values(by=['ticker','date'])
trainSet.head()