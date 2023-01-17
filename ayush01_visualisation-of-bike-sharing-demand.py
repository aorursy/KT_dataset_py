import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sn

from scipy import stats

#import missingno as msno

from datetime import datetime

import matplotlib.pyplot as plt

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
daily_Data = pd.read_csv("../input/train.csv")

daily_Data.head()
daily_Data.shape
daily_Data.dtypes
daily_Data.columns
daily_Data.isnull().sum()
print('season:',daily_Data.season.unique())

print("holiday",daily_Data.holiday.unique())

print('workingday:',daily_Data.workingday.unique())

print('weather:',daily_Data.weather.unique())

print('temp:',daily_Data.temp.unique())

print('atemp:',daily_Data.atemp.unique())

print('humidity:',daily_Data.humidity.unique())
from collections import Counter

Counter(daily_Data["holiday"])
#Data visualization
sn.barplot(x='season', y='count', data=daily_Data)
f, ax = plt.subplots(figsize=(5,5))

plt.hist(x="season", data=daily_Data, color="c");

plt.xlabel("season")
daily_Data.season.value_counts()
f, ax = plt.subplots(figsize=(5,5))

plt.hist(x="holiday", data=daily_Data,color='c');

plt.xlabel("holiday")
daily_Data.holiday.value_counts()
f, ax = plt.subplots(figsize=(5,5))

plt.hist(x="workingday",data=daily_Data,color='c');

plt.xlabel("workingday")
daily_Data.workingday.value_counts()
f, ax = plt.subplots(figsize=(5,5))

plt.hist(x="weather",data=daily_Data,color='c');

plt.xlabel("weather")
daily_Data.weather.value_counts()
plt.hist(x="temp",data=daily_Data,edgecolor="black",linewidth=2)
#correlation

corrMatt = daily_Data[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
season=pd.get_dummies(daily_Data['season'])

daily_Data=pd.concat([daily_Data,season],axis=1)
weather=pd.get_dummies(daily_Data['weather'])

daily_Data=pd.concat([daily_Data,weather],axis=1)
daily_Data.head()
daily_Data.shape
daily_Data=daily_Data.drop("season",axis=1)

daily_Data=daily_Data.drop("weather",axis=1)

daily_Data=daily_Data.drop("casual",axis=1)

daily_Data=daily_Data.drop("registered",axis=1)

labels=daily_Data.pop("count")
daily_Data.head()
daily_Data.shape
labels.head()
daily_Data["hour"] = [t.hour for t in pd.DatetimeIndex(daily_Data.datetime)]

daily_Data["day"] = [t.dayofweek for t in pd.DatetimeIndex(daily_Data.datetime)]

daily_Data["month"] = [t.month for t in pd.DatetimeIndex(daily_Data.datetime)]

daily_Data['year'] = [t.year for t in pd.DatetimeIndex(daily_Data.datetime)]

daily_Data['year'] = daily_Data['year'].map({2011:0, 2012:1})
daily_Data.head()
print("holiday",daily_Data.holiday.unique())

print('workingday:',daily_Data.workingday.unique())

print('temp:',daily_Data.temp.unique())

print('atemp:',daily_Data.atemp.unique())

print('hour:',daily_Data.hour.unique())

print('day:',daily_Data.day.unique())

print('month:',daily_Data.month.unique())

print('year:',daily_Data.year.unique())
daily_Data=daily_Data.drop("datetime",axis=1)
daily_Data.columns
from sklearn.cross_validation import  train_test_split

X_train, X_test, y_train, y_test = train_test_split(daily_Data, labels, test_size=.3, random_state=42)
X_train.head()
#from sklearn.ensemble import RandomForestRegressor

#forest = RandomForestRegressor(n_estimators = 400, criterion='mse',random_state=1, n_jobs=-1)

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators = 400, criterion='mse',random_state=1, n_jobs=-1)

reg=reg.fit(X_train, y_train)

feat=(reg.feature_importances_)

y_train_pred = reg.predict(X_train)

y_test_pred = reg.predict(X_test)

print(feat)

plt.hist(feat,bins=30)
from sklearn.metrics import mean_squared_error, r2_score

#Root_Mean_Square_Log_Error(RMSE) is accuracy criteria for this problem

print('RMSLE train: %.3f' % np.sqrt(mean_squared_error(np.log(y_train + 1), np.log(y_train_pred + 1))))

print('RMSLE test: %.3f' % np.sqrt(mean_squared_error(np.log(y_test + 1), np.log(y_test_pred + 1))))

print('R2 train: %.3f' % r2_score(y_train, y_train_pred))

print('R2 test: %.3f' % r2_score(y_test, y_test_pred))