# This is a work in progress on avocado price investigation.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/avocado.csv", parse_dates=['Date'])
# assume vol-diff and bag-diff columns are 0, this is not true for the numeric lookup codes?
# I would expect zero influence of any of the bags columns on the price, just a way of packaging?
df['year'] = df['Date'].apply(lambda x: x.year)
df['month'] =df['Date'].apply(lambda x: x.month)
df.head()
avc = df.sort_values('Date').groupby(['Date']).agg({'AveragePrice': 'mean','Total Volume': 'sum','4046': 'sum', '4225': 'sum', '4770': 'sum'}).reset_index(level=0)
#get month and date out in separate column.
avc['year'] = pd.DatetimeIndex(avc['Date']).year
avc['month'] = pd.DatetimeIndex(avc['Date']).month

sns.set_context('notebook', font_scale=1)
sns.set_style('ticks')
sns.lmplot(x='Total Volume',y='AveragePrice', data=avc, hue='year')
plt.title('vol - price')
sns.set_context('notebook', font_scale=1)
sns.set_style('ticks')
sns.violinplot(x='type',y='AveragePrice', data=df, hue='year')
plt.title('vol - price')
#investigate the bags,
#I expect lineair relation between the 3 bags sizes and total volume
# TotalVolume = x1 * smallbag + x2 * largebag + x3 * XLarge Bags.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
linReg = LinearRegression()
X=df[['Small Bags', 'Large Bags', 'XLarge Bags']]
y=df[['Total Volume']]
linReg.fit(X=X, y = y)
y_pred = linReg.predict(X)
(r2_score(y, y_pred), linReg.coef_)
# the r2 seems to be reasonable, the coeff is way off my expectations, x1 > x2 >> x3 and x3 is negative!
#Apply some ensemble models, start with gradientboosting
from sklearn.ensemble import GradientBoostingRegressor

y = df['AveragePrice']
df2 = df.drop(['Small Bags', 'Large Bags', 'XLarge Bags','AveragePrice','Date'], axis=1)
df2 = pd.get_dummies(data=df2, columns=['region', 'type'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    df2, y, test_size=0.33)
gradModel = GradientBoostingRegressor(learning_rate=0.15, max_depth=4)
gradModel.fit(X_train,y_train)
y_pred=gradModel.predict(X_test)
(r2_score(y_train, gradModel.predict(X_train)),r2_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score
import sklearn.metrics
scores = cross_val_score(gradModel, df2, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))