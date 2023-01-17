%matplotlib inline
import pandas as pd

from pandas import Series

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import date, datetime, timedelta

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
train_df = pd.read_csv("../input/train.csv")

store_df = pd.read_csv("../input/store.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.Date = train_df.Date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())

train_df[u'Year'] = train_df.Date.apply(lambda x: x.year)

train_df[u'Month'] = train_df.Date.apply(lambda x: x.month)

train_df = train_df.drop([u'Date', u'Customers'], axis=1)

day_dummies_result = pd.get_dummies(train_df[u'DayOfWeek'], prefix=u'Day')

train_df = train_df.join(day_dummies_result)

train_df = train_df.drop([u'DayOfWeek'], axis=1)

train_df["StateHoliday"] = train_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
train_df.head()
store_df = store_df.drop(['Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'], axis=1)

assortment_dummies_result = pd.get_dummies(store_df[u'Assortment'], prefix=u'Assortment')

store_type_dummies_result = pd.get_dummies(store_df[u'StoreType'], prefix=u'StoreType')

store_df = store_df.join(assortment_dummies_result).join(store_type_dummies_result)

store_df = store_df.drop([u'Assortment'], axis=1)

store_df = store_df.drop([u'StoreType'], axis=1)
store_df.head()
#store_df.CompetitionDistance = store_df.CompetitionDistance.fillna(float('inf'))

#store_df.CompetitionDistance = 1 / (store_df.CompetitionDistance + 1)

store_df.CompetitionDistance = store_df.CompetitionDistance.fillna(store_df.CompetitionDistance.max())
train_df.info()

print("----------------------------")

store_df.info()

print("----------------------------")

test_df.info()
train = pd.merge(train_df, store_df, on='Store')
train['CompetitionTime'] = (train['Year'] - train['CompetitionOpenSinceYear']) * 12 + (train['Month'] - train['CompetitionOpenSinceMonth'])

#train.CompetitionTime = train.CompetitionTime.fillna(float('inf'))

#train['CompetitionTime'] = 1 / (train['CompetitionTime'] + 1)

train.CompetitionTime = train.CompetitionTime.fillna(train.CompetitionTime.max())
train = train.drop(['Year', 'Month', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'], axis=1)
X_train = train.drop(["Sales","Store"],axis=1)

Y_train = train["Sales"]

model = RandomForestRegressor(n_estimators=30, n_jobs=2)

model.fit(X_train, Y_train)
test_df.Date = test_df.Date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())

test_df[u'Year'] = test_df.Date.apply(lambda x: x.year)

test_df[u'Month'] = test_df.Date.apply(lambda x: x.month)

test_df = test_df.drop([u'Date'], axis=1)

day_dummies_result = pd.get_dummies(test_df[u'DayOfWeek'], prefix=u'Day')

test_df = test_df.join(day_dummies_result)

test_df = test_df.drop([u'DayOfWeek'], axis=1)

test_df["StateHoliday"] = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})