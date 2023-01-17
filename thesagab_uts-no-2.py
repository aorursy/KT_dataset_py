# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

from time import time





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/dataset_fixed.csv')

df.head()
df.info()
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
df = df.dropna()

df.isnull().sum()
df.info()
df.describe()
df['hours'] = df['timestamp'].dt.hour

df = df.drop(['id', 'timestamp'], axis=1)

df.head()
sns.heatmap(df.corr(), annot=True)
for column in df.columns:

    q1 = df[column].quantile(0.25)

    q3 = df[column].quantile(0.75)

    iqr = q3-q1 #Interquartile range

    fence_low  = q1-1.5*iqr

    fence_high = q3+1.5*iqr

    df = df.loc[(df[column] > fence_low) & (df[column] < fence_high)]
df.describe()
df.hist(bins=30, figsize=(15,8))
sns.heatmap(df.corr(), annot=True)
dummies = pd.get_dummies(df["node"], prefix="node")

df = pd.concat([df.drop("node", axis=1), dummies], axis=1)

df.head()
std_scaler = StandardScaler()

df_numeric = df.drop(["node_0", "node_1", "node_9"], axis=1)

df_numeric_scaled = pd.DataFrame(std_scaler.fit_transform(df_numeric), columns=df_numeric.columns)

df_numeric_scaled.head()
df_category = df.drop(['co2', 'temp', 'humidity', 'light', 'hours'], axis=1).reset_index().drop(['index'], axis=1)

df_category.head()
df_new = pd.concat([df_numeric_scaled, df_category], sort=False, axis=1)

df_new.head()
X = df.drop(['humidity'], axis=1)

y = df['humidity']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)

print("Ukuran data training {}, data testing {}".format(X_train.shape, X_test.shape))
linreg = LinearRegression()

time_start = time()

linreg.fit(X_train, y_train)

time_end = time()

accuracy = linreg.score(X_test, y_test)

print('LinearRegression accuracy:', accuracy)

print('Time:', time_end-time_start)
ridged = Ridge()

time_start = time()

linreg.fit(X_train, y_train)

time_end = time()

accuracy = linreg.score(X_test, y_test)

print('Ridge accuracy:', accuracy)

print('Time:', time_end-time_start)
lasso = Lasso()

time_start = time()

lasso.fit(X_train, y_train)

time_end = time()

accuracy = lasso.score(X_test, y_test)

print('Lasso accuracy:', accuracy)

print('Time:', time_end-time_start)
bayesian = BayesianRidge()

time_start = time()

bayesian.fit(X_train, y_train)

time_end = time()

accuracy = bayesian.score(X_test, y_test)

print('BayesianRidge accuracy:', accuracy)

print('Time:', time_end-time_start)
dtr = DecisionTreeRegressor()

time_start = time()

dtr.fit(X_train, y_train)

time_end = time()

accuracy = dtr.score(X_test, y_test)

print('DecisionTreeRegressor accuracy:', accuracy)

print('Time:', time_end-time_start)
rf = RandomForestRegressor()

time_start = time()

rf.fit(X_train, y_train)

time_end = time()

accuracy = rf.score(X_test, y_test)

print('RandomForestRegressor accuracy:', accuracy)

print('Time:', time_end-time_start)