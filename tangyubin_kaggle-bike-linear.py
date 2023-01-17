# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import missingno as msno

import matplotlib.pyplot as plt

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_origin = pd.read_csv('../input/train.csv')
data_origin.head(2)
data_origin.describe()
data_origin.info()
data_origin.nunique()
data_origin.dtypes[data_origin.dtypes == 'int64'].index
cat_feature = ['season', 'holiday', 'workingday', 'weather']

data_train  = data_origin

data_train[cat_feature] = data_train[cat_feature].astype('category')

data_train.head()
data_train.info()
dt_idx = pd.DatetimeIndex(data_train['datetime'])

data_train['hour'] = dt_idx.hour

data_train['dayofweek'] = dt_idx.dayofweek

data_train['month'] = dt_idx.month
data_train[['hour', 'dayofweek', 'month']] = data_train[['hour', 'dayofweek', 'month']].astype('category')

data_train.info()
data_train = data_train.drop('datetime', axis=1)
data_train.head(2)
data_train = data_train.drop(['casual', 'registered'], axis=1)
data_train.head(2)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

sns.boxplot(data=data_train, y='count', ax=axes[0][0])

sns.boxplot(data=data_train, y='count', x='season', ax=axes[0][1])

sns.boxplot(data=data_train, y='count', x='hour', ax=axes[1][0])

sns.boxplot(data=data_train, y='count', x='workingday', ax=axes[1][1])
data_train_without_outliers = data_train[np.abs(data_train['count'] - 

                                        data_train['count'].mean())

                                        < data_train['count'].std() * 3]
print('before:', data_train.shape)

print('after:', data_train_without_outliers.shape)
data_train.dtypes[data_train.dtypes != 'category'].index
corr_matt = data_train[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr()

corr_matt
sns.heatmap(corr_matt, vmax=0.8, annot=True, cmap="YlGnBu")
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 5))

sns.regplot(x='temp', y='count', data=data_train, ax=ax1)

sns.regplot(x='humidity', y='count', data=data_train, ax=ax2)

sns.regplot(x='windspeed', y='count', data=data_train, ax=ax3)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

sns.distplot(data_train['count'], ax=axes[0][0])

stats.probplot(data_train['count'], dist='norm', fit=True, plot = axes[0][1])

sns.distplot(data_train_without_outliers['count'].map(np.log), ax=axes[1][0])

stats.probplot(data_train_without_outliers['count'].map(np.log1p), dist='norm', fit=True, plot = axes[1][1])

data_train.head()
month_agg = data_train.groupby("month").mean().reset_index()

month_agg
sns.barplot(data=month_agg, x='month', y='count')
hour_week_agg = data_train.groupby(['hour', 'dayofweek']).mean().reset_index()

hour_week_agg.head()
sns.pointplot(data=hour_week_agg, x='hour', y='count', hue='dayofweek')
hour_season_agg = data_train.groupby(['hour', 'season']).mean().reset_index()

hour_season_agg.head()
sns.pointplot(data=hour_season_agg, x='hour', y='count', hue='season')
data_train = pd.read_csv('../input/train.csv', index_col='datetime', parse_dates=True)

data_test = pd.read_csv('../input/test.csv', index_col='datetime', parse_dates=True)
data_train.shape, data_test.shape
data = data_train.append(data_test, sort=True)

data.shape
data['hour'] = data.index.hour

data['dayofweek'] = data.index.dayofweek

data['month'] = data.index.month

data['year'] = data.index.year
from sklearn.ensemble import RandomForestRegressor

data_wind0 = data[data['windspeed'] == 0]

data_windnot0 = data[data['windspeed'] != 0]

wind_columns = ['atemp', 'year', 'humidity', 'season', 'temp', 'weather', 'month']



rfr_wind = RandomForestRegressor(n_estimators=10)

rfr_wind.fit(data_windnot0[wind_columns], data_windnot0['windspeed'])

data_wind0.loc[:,'windspeed'] = rfr_wind.predict(data_wind0[wind_columns])

data = data_windnot0.append(data_wind0)
data.head(2)
data.nunique()
category_columns = ['holiday', 'season', 'weather', 'workingday', 'hour', 'dayofweek',\

                   'month', 'year']
df
df