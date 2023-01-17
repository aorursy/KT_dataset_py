import pandas as pd

import numpy as np

import xgboost as xgb

import seaborn as sns

from sklearn.metrics import mean_squared_error



# Air Pollution HackerEarth Challenge



train = pd.read_csv('../input/hackerearth-air-pollution-index/train.csv')

test = pd.read_csv('../input/hackerearth-air-pollution-index/test.csv',  parse_dates=['date_time'])

train.head()

train.drop(columns=['date_time'])
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=3, figsize=(10,5))

sns.boxplot(x=train['clouds_all'], ax = axs[0])

sns.boxplot(x=train['traffic_volume'], ax = axs[1])

sns.boxplot(x=train['temperature'], ax = axs[2])

plt.show()
Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3- Q1

print(IQR, train.shape)
train = train[~((train < (Q1 - 5*IQR)) | (train > (Q3 + 5*IQR))).any(axis=1)]

train = train.dropna()

train.shape
train.describe()
# Categorical Variables



train.loc[train.is_holiday == 'None', 'is_holiday'] = 0

train.loc[train.is_holiday != 0, 'is_holiday'] = 1

train['is_holiday'] = train['is_holiday'].astype('category')

train['is_holiday'] = train['is_holiday'].cat.codes

train['weather_type'] = train['weather_type'].astype('category')

train['weather_type'] = train['weather_type'].cat.codes
print(train['weather_type'].unique())
train.head()
f, ax = plt.subplots(figsize=(11, 9))

corr = train.corr()

sns.heatmap(corr, annot=False)

corr

plt.show()