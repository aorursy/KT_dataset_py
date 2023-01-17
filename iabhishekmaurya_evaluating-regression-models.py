# Importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
# Reading the Dataset

train_data = pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')

train_data.head()
train_data.shape
train_data.describe()
train_data.isnull().sum()
train_data.info()
train_data.columns
train_data = train_data.iloc[:,3:9]

train_data.head()
train_data['Radiation'].describe()
f, ax = plt.subplots(figsize=(15,8))

sns.distplot(train_data['Radiation'])

plt.xlim([-10,1602])
plt.figure(figsize=(10,10))

sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn')

plt.show()
X = train_data.iloc[:,1:]

y = train_data.iloc[:,0]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 25)
from sklearn.linear_model import LinearRegression

regresor= LinearRegression()

regresor.fit(X_train, y_train)

regresor_pred = regresor.predict(X_test)
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()

rf_reg.fit(X_train, y_train)

randomforest_pred= rf_reg.predict(X_test)
from sklearn.metrics import explained_variance_score

print(explained_variance_score(y_test, regresor_pred))

print(explained_variance_score(y_test, randomforest_pred))
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, regresor_pred))

print(mean_absolute_error(y_test, randomforest_pred))
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, regresor_pred))

print(mean_squared_error(y_test, randomforest_pred))
from sklearn.metrics import median_absolute_error

print(median_absolute_error(y_test, regresor_pred))

print(median_absolute_error(y_test, randomforest_pred))
from sklearn.metrics import r2_score

print(r2_score(y_test, regresor_pred))

print(r2_score(y_test, randomforest_pred))