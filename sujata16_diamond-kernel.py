# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import collections

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

 #       print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

df  = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')

print(list(df.columns))

print(df.head(3))

plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')

#sns.pairplot(df[df.columns], height=3)

plt.show()
# feature engineering -- manual 

import collections

diamond_cut = {'Fair':0,

               'Good':1,

               'Very Good':2, 

               'Premium':3,

               'Ideal':4}



diamond_color = {'J':0,

                 'I':1, 

                 'H':2,

                 'G':3,

                 'F':4,

                 'E':5,

                 'D':6}



diamond_clarity = {'I1':0,

                   'SI2':1,

                   'SI1':2,

                   'VS2':3,

                   'VS1':4,

                   'VVS2':5,

                   'VVS1':6,

                   'IF':7}

#

df.cut  = df.cut.map(diamond_cut)

df.color = df.color.map(diamond_color)

df.clarity = df.clarity.map(diamond_clarity)

print(df.head())

plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')

plt.show()
# let's look at the distribution of various columns now. 

# Knowing distribution helps in the modeling

plt.figure(figsize=(10,7))

sns.distplot(df.carat)

plt.grid()

plt.show()
plt.figure(figsize=(10,7))

sns.distplot(df.price)

plt.grid()

plt.show()
plt.figure(figsize=(10,7))

sns.distplot(df.x)

plt.grid()

plt.show()
plt.figure(figsize=(10,7))

sns.distplot(df.y)

plt.grid()

plt.show()
plt.figure(figsize=(10,7))

sns.distplot(df.z)

plt.grid()

plt.show()
from sklearn.linear_model import LinearRegression, Ridge 

from sklearn.model_selection import train_test_split

#

X = df.drop(['price'],1)

y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#

# scaling the entire data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

print(X_train)

#

print("Linear Regression")

lr = LinearRegression()

lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))

# 

print("Ridge Regression")

rr = Ridge()

rr.fit(X_train, y_train)

print(rr.score(X_test, y_test))

#

print("Ridge Regression normalized")

rr_norm = Ridge(normalize=True)

rr_norm.fit(X_train, y_train)

print(rr_norm.score(X_test, y_test))

#

print("testing the gradient boosting model")

from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()

gb.fit(X_train, y_train)

print(gb.score(X_test, y_test))