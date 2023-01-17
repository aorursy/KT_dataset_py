import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb



dir_ = '/kaggle/input/ashrae-energy-prediction/' 
train = pd.read_csv(dir_+'train.csv')

weather_train = pd.read_csv(dir_+'weather_train.csv')

test = pd.read_csv(dir_+'test.csv')

weather_test = pd.read_csv(dir_+'weather_test.csv')

print(train.shape)

train.head()
print(weather_train.shape)

weather_train.head()
train[['meter_reading']].describe()
plt.style.use('seaborn')
meter_values = train['meter'].unique()

ax = sns.barplot(meter_values, [train[train['meter']==meter_values[i]].count()[0] for i in range(len(meter_values))])

ax.set(xlabel='meter')

plt.show()
cloud_coverage_values = weather_train['cloud_coverage'].unique()

ax = sns.barplot(cloud_coverage_values, [weather_train[weather_train['cloud_coverage']==cloud_coverage_values[i]].count()[0] for i in range(len(cloud_coverage_values))])

ax.set(xlabel='cloud_coverage')

plt.show()
sns.distplot(weather_train['air_temperature'].dropna())
sns.distplot(weather_train['dew_temperature'].dropna())
sns.distplot(weather_train['sea_level_pressure'].dropna())
sns.distplot(weather_train['wind_speed'].dropna(), hist=False)