import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LinearRegression

import seaborn as sns

from sklearn import metrics

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))

days = pd.read_csv('../input/bike-sharing-dataset/day.csv', index_col=0)

hours = pd.read_csv('../input/bike-sharing-dataset/hour.csv', index_col=0)
df = pd.concat([days,hours])
df.drop(labels='dteday', axis=1)
plt.hist(df.cnt, bins=3)
df.corr()['cnt']
df = pd.get_dummies(df)

df = df.fillna(value=0)

x = df.drop(labels=['cnt','casual','registered'],axis=1)

y = df['cnt']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=17)
lr = LinearRegression()

lr.fit(x_train,y_train)

predictions = lr.predict(x_test)

print('MAE= ', metrics.mean_absolute_error(y_test,predictions))

print('MSE= ', metrics.mean_squared_error(y_test,predictions))

print('RMS= ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
reg = DecisionTreeRegressor(min_samples_leaf=5, random_state=17)

reg.fit(x_train,y_train)

predictions = reg.predict(x_test)

print('MAE= ', metrics.mean_absolute_error(y_test,predictions))

print('MSE= ', metrics.mean_squared_error(y_test,predictions))

print('RMS= ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
reg = DecisionTreeRegressor(min_samples_leaf=10, random_state=17)

reg.fit(x_train,y_train)

predictions = reg.predict(x_test)

print('MAE= ', metrics.mean_absolute_error(y_test,predictions))

print('MSE= ', metrics.mean_squared_error(y_test,predictions))

print('RMS= ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
reg = DecisionTreeRegressor(min_samples_leaf=15, random_state=17)

reg.fit(x_train,y_train)

predictions = reg.predict(x_test)

print('MAE= ', metrics.mean_absolute_error(y_test,predictions))

print('MSE= ', metrics.mean_squared_error(y_test,predictions))

print('RMS= ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
# As I've learned, randomforest performed considerably better

reg = RandomForestRegressor(min_samples_leaf=5, random_state=17)

reg.fit(x_train, y_train)

predictions = reg.predict(x_test)

print('MAE= ', metrics.mean_absolute_error(y_test,predictions))

print('MSE= ', metrics.mean_squared_error(y_test,predictions))

print('RMS= ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
reg = RandomForestRegressor(min_samples_leaf=5, random_state=17, n_estimators=80)

reg.fit(x_train, y_train)

predictions = reg.predict(x_test)

print('MAE= ', metrics.mean_absolute_error(y_test,predictions))

print('MSE= ', metrics.mean_squared_error(y_test,predictions))

print('RMS= ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))