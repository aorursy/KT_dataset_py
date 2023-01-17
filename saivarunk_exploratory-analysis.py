import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import csv as csv

import xgboost as xgb

import seaborn as sns



from scipy import stats, integrate

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from catboost import Pool, CatBoostRegressor

from sklearn import svm

from sklearn import metrics



import warnings



pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore")

%matplotlib inline



sns.set(color_codes=True)

np.random.seed(sum(map(ord, "regression")))
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
# verify shape of test and train dataframes

print(train_data.shape)

print(test_data.shape)
train_data.head()
test_data[0:10]
# split datetime into year, month, hour, day features

date = pd.DatetimeIndex(train_data['datetime'])

train_data['year'] = date.year

train_data['month'] = date.month

train_data['date'] = date.date

train_data['hour'] = date.hour

train_data['weekday'] = date.weekday

 

date = pd.DatetimeIndex(test_data['datetime'])

test_data['year'] = date.year

test_data['date'] = date.date

test_data['month'] = date.month

test_data['hour'] = date.hour

test_data['weekday'] = date.weekday
train_data.head()
train_data["season"] = train_data.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })

train_data["weather"] = train_data.weather.map({1: "Clear + Few clouds",\

                                        2 : " Mist + Cloudy", \

                                        3 : " Light Snow, Light Rain", \

                                        4 :" Heavy Rain + Ice Pallets" })
train_data.head()
sns.stripplot(x="season", y="count", data=train_data, jitter=True)
sns.stripplot(x="season", y="count", data=train_data, jitter=True)
sns.stripplot(x="hour", y="count", data=train_data, jitter=True)
weather = sns.stripplot(x="year", y="count", data=train_data, jitter=True)
season = sns.stripplot(x="season", y="count", data=train_data, jitter=True)
# facet plot for comparing relation between hour and count split by seasons

sns.factorplot(x="hour", y="count", col="season", hue="holiday", data=train_data);
# linear regression for temp vs count

sns.regplot(x="temp", y="count", data=train_data);
sns.lmplot(x="temp", y="count", row="season",

               truncate=True, size=5, data=train_data)
# linear regression temprature vs count for seasons

sns.lmplot(x="count", y="temp", col="season", data=train_data,

           col_wrap=2, size=3);
# correlation heatmap

sns.clustermap(train_data.corr(), center=0, linewidths=.75, figsize=(13, 13), vmax=.8, square=True,annot=True)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# split datetime into year, month, hour, day features

date = pd.DatetimeIndex(train['datetime'])

train['year'] = date.year

train['month'] = date.month

train['date'] = date.date

train['hour'] = date.hour

train['weekday'] = date.weekday

train = train.drop('datetime', axis = 1)

 

date = pd.DatetimeIndex(test['datetime'])

test['year'] = date.year

test['date'] = date.date

test['month'] = date.month

test['hour'] = date.hour

test['weekday'] = date.weekday

test = test.drop('datetime', axis = 1)
train.head()
test.head()
y_train = train['count']

x_train = train.drop(['date', 'registered', 'casual', 'count', 'month', 'hour', 'season', 'holiday'], axis=1)

x_test = test.drop(['date', 'month', 'hour', 'season', 'holiday'], axis=1)

print(x_train.shape)

print(x_test.shape)
x_train.head()
x_test.head()
categoricalFeatureNames = ["workingday","weather","weekday", "year"]

for var in categoricalFeatureNames:

    train[var] = train[var].astype("category")

    

train.info()
X_train, X_test, y_train, y_test = train_test_split(train.drop(['date', 'registered', 'casual', 'count'], axis=1), 

                                                    train['count'], test_size=0.2, random_state=42)
gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01)

gbm.fit(X_train,y_train)

preds = gbm.predict(X_test)

gbm.score(X_test, y_test)
sns.regplot(y_test, preds, fit_reg = False)
fig, (ax0, ax1) = plt.subplots(ncols=2)

sns.distplot(y_test, kde=False, fit=stats.gamma, ax=ax0, hist=False, rug=True);

sns.distplot(preds, kde=False, fit=stats.gamma, ax=ax1, hist=False, rug=True);
cat = CatBoostRegressor(iterations=500, learning_rate=0.25, depth=10)

cat.fit(X_train,y_train)

preds = cat.predict(X_test)