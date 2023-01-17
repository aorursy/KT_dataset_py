# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

data.head()
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[['weather','temp','humidity', 'windspeed']], data['count'], test_size=0.33, shuffle=False)
model = LinearRegression()

model.fit(X_train, y_train)

predict = model.predict(X_test)
predict
from sklearn.metrics import mean_squared_error

from math import sqrt



sqrt(mean_squared_error(y_test, predict))
data['datetime'] = pd.to_datetime(data['datetime'])

data['weekday'] = data['datetime'].dt.dayofweek

data['season'] = data['datetime'].apply(lambda x: (x.month % 12 + 3) // 3)
data.head()
from sklearn.preprocessing import OneHotEncoder



encoder = OneHotEncoder(categories='auto', sparse=False)

X_categorical = encoder.fit_transform(data[['weekday', 'season', 'weather']])

X_usual = data[['temp', 'humidity', 'windspeed']]
X_categorical.shape
X_usual.shape
from numpy import hstack



X = hstack([X_usual, X_categorical])

X_train, X_test, y_train, y_test = train_test_split(X, data['count'], test_size=0.33, shuffle=False)

model = LinearRegression()

model.fit(X_train, y_train)

predict = model.predict(X_test)

sqrt(mean_squared_error(y_test, predict))
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)

predict = model.predict(X_test)

sqrt(mean_squared_error(y_test, predict))
data[['weekday', 'season', 'weather']] = data[['weekday', 'season', 'weather']].astype('object')

X_train, X_test, y_train, y_test = train_test_split(data[['weekday', 'season', 'weather', 'temp', 'humidity', 'windspeed']], data['count'], test_size=0.33, shuffle=False)
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, y_train)

predict = model.predict(X_test)

sqrt(mean_squared_error(y_test, predict))