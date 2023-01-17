# Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

from scipy.stats import norm

import warnings

import datetime

import time
#import data file for kaggle

train = pd.read_csv('../input/game-rating/train.csv')

test = pd.read_csv('../input/game-rating/test.csv')
train.head()

test.head()

train.shape, test.shape
train.info() ,

print('\n'),

test.info()
train.select_dtypes(include=object).head()
test.select_dtypes(include=object).head()
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');
train.isnull().sum()
test.isnull().sum()
x =pd.isnull(train['game_released'])

train[x]
x =pd.isnull(test['game_released'])

test[x]
c =train['game_released'].replace(np.nan , 0, inplace=True )
cc =test['game_released'].replace(np.nan , 0, inplace=True )
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))



# train data 

sns.heatmap(train.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')

ax[0].set_title('Train data')



# test data

sns.heatmap(test.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')

ax[1].set_title('Test data');