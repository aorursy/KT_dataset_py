# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
train.dtypes
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
train['pickup_hour'] = train['pickup_datetime'].dt.hour

train['pickup_minute'] = train['pickup_datetime'].dt.minute

train['pickup_day'] = train['pickup_datetime'].dt.day

train['pickup_month'] = train['pickup_datetime'].dt.month

# Just in case, let's get the second of the pickup time as well

train['pickup_second'] = train['pickup_datetime'].dt.second
train.head()
sns.distplot(train['pickup_month'])
sns.distplot(train['pickup_day'])
sns.distplot(train['pickup_hour'])
sns.distplot(train['pickup_minute'])
sns.distplot(train['pickup_second'])
sns.distplot(train['trip_duration'])
train['trip_duration'].describe()
sns.distplot(train['passenger_count'])
train['passenger_count'].describe()
train = train[train['trip_duration'] < 500000]

sns.distplot(train['trip_duration'])
train_below_5000 = train[train['trip_duration'] < 5000]

sns.distplot(train_below_5000['trip_duration'])
sns.heatmap(train.drop(['id','dropoff_datetime', 'pickup_datetime', 'store_and_fwd_flag'], axis=1).corr())
from sklearn.ensemble import RandomForestRegressor
train.head()


def encode(X):

    if X == 'Y':

        return 1

    else:

        return 0

    

train['store_and_fwd_flag'] = train['store_and_fwd_flag'].apply(lambda x: encode(x))

train.head()
train_trim = train.drop(['id','dropoff_datetime', 'pickup_datetime', 'store_and_fwd_flag'], axis=1)
# I have chosen these parameters to fit the Kaggle limits on kernel run time

rf_regressor = RandomForestRegressor(n_estimators=10, max_features='sqrt', n_jobs=-1)

X = train_trim.drop('trip_duration', axis=1)

y = train_trim['trip_duration']
from sklearn.model_selection import train_test_split

from random import randint

from sklearn import metrics

for fold in range(3):

    print('Testing model for fold {} ...'.format(fold + 1))

    randnum = randint(1, 102)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=randnum)

    rf_regressor.fit(X_train, y_train)

    predictions = rf_regressor.predict(X_test)

    print('Results for fold {}'.format(fold + 1))

    print('Mean squared logarithmic error: {}'.format(metrics.mean_squared_log_error(predictions, y_test)))