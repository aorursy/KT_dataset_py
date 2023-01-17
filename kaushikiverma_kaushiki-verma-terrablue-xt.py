#!/usr/bin/env python

# coding: utf-8

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('../input/into-the-future/train.csv')

dataset1 = pd.read_csv('../input/into-the-future/test.csv')

dataset.shape

dataset.describe()

x= dataset['feature_1'].values

y= dataset['feature_2'].values

plt.plot(x,y)

plt.xlabel('feature_1')

plt.ylabel('feature_2')

plt.show()

x=dataset['feature_1'].values.reshape(-1,1)

y=dataset['feature_2'].values.reshape(-1,1)

x1=dataset1['feature_1'].values.reshape(-1,1) 

y1=dataset['feature_2'].values.reshape(-1,1)

regressor=LinearRegression()

regressor.fit(x,y)

print(regressor.intercept_)

print(regressor.intercept_)

print(regressor.coef_)

y_pred=regressor.predict(x1)

df=pd.DataFrame({'Predicted':y_pred.flatten()})

print(df.head())

df.to_csv('solution.csv')