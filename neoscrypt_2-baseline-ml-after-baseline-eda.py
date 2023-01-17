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
baseline=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
baseline.head()
dt_idx = pd.DatetimeIndex(baseline['datetime'])
baseline['year'] = dt_idx.year
baseline['month'] = dt_idx.month
baseline['day'] = dt_idx.day
baseline['dayofweek'] = dt_idx.dayofweek
baseline['hour'] = dt_idx.hour
baseline['minute'] = dt_idx.minute
baseline['second'] = dt_idx.second
print(baseline.shape)
baseline.head()
baseline['weekend'] = 0
baseline.loc[baseline['dayofweek'] >= 5, 'weekend'] = 1

df = baseline.copy()
df.head()
interested = ['year', 'month', 'day', 'hour', 'workingday']
label = 'count'
train, test = df[:7000], df[7000:]
X_train, X_test = train[interested], test[interested]
y_train, y_test = train[label], test[label]
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=20, n_jobs=-1)
model
from sklearn.model_selection import cross_val_score

score = cross_val_score(model, X_train, y_train, cv=20).mean()

print("Score = {0:.5f}".format(score))
def df_split(df, interested):
    label = 'count'
    train, test = df[:7000], df[7000:]
    X_train, X_test = train[interested], test[interested]
    y_train, y_test = train[label], test[label]
    return X_train, X_test, y_train, y_test
# add season info
interested = ['year', 'month', 'day', 'hour', 'workingday', 'season']
X_train, X_test, y_train, y_test = df_split(df, interested)

score = cross_val_score(model, X_train, y_train, cv=20).mean()

print("Score = {0:.5f}".format(score))
# del season info, add dayofweek
interested = ['year', 'month', 'day', 'hour', 'workingday', 'dayofweek']
X_train, X_test, y_train, y_test = df_split(df, interested)

score = cross_val_score(model, X_train, y_train, cv=20).mean()

print("Score = {0:.5f}".format(score))
# add weekend info
interested = ['year', 'month', 'day', 'hour', 'workingday', 'dayofweek', 'weekend']
X_train, X_test, y_train, y_test = df_split(df, interested)

score = cross_val_score(model, X_train, y_train, cv=20).mean()

print("Score = {0:.5f}".format(score))
# verify feature importances
model.fit(X_train, y_train)
model.feature_importances_
X_train.columns
sorted(zip(model.feature_importances_, X_train.columns))
# hour seems the most informative of all(among datetime)






