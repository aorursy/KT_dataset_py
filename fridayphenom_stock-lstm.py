import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas import datetime

import math, time

import itertools

from sklearn import preprocessing

import datetime

from operator import itemgetter

from sklearn.metrics import mean_squared_error

from math import sqrt

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.recurrent import LSTM

from keras.models import load_model

import keras

import h5py

import requests

import os
stock_name = 'YHOO'



prices_dataset = pd.read_csv('../input/prices-split-adjusted.csv')

prices_dataset['adj close'] = prices_dataset.close

prices_dataset.drop(['close'], 1, inplace=True)

stock_df = prices_dataset[prices_dataset['symbol'] == stock_name]

print(stock_df.head())

min_max_scaler = preprocessing.MinMaxScaler()

stock_df['open'] = min_max_scaler.fit_transform(stock_df.open.values.reshape(-1, 1))

stock_df['high'] = min_max_scaler.fit_transform(stock_df.high.values.reshape(-1, 1))

stock_df['low'] = min_max_scaler.fit_transform(stock_df.low.values.reshape(-1, 1))

stock_df['volume'] = min_max_scaler.fit_transform(stock_df.volume.values.reshape(-1, 1))

stock_df['adj close'] = min_max_scaler.fit_transform(stock_df['adj close'].values.reshape(-1, 1))

print(stock_df.head())

amount_of_features = len(stock)