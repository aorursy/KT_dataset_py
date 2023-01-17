# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sb
startups = pd.read_csv('../input/sp-startup/50_Startups.csv') 

df = startups
df.head(5)
df.info()
df.shape

df.isnull().sum()
df.corr()
sb.heatmap(df.corr());
sb.scatterplot(x="R&D Spend", y="Profit", data = df)
sb.distplot(df["R&D Spend"], bins=16, color="blue");
df.describe().T
df["State"].unique() 
df['State'] = pd.Categorical(df['State'])

dfDummies = pd.get_dummies(df['State'])

dfDummies
df = pd.concat([df, dfDummies], axis=1)

df = df.drop(['State', "California"], axis = 1)

df.head()
X = df.drop("Profit", axis = 1)

y = df["Profit"]
X.head(5)
y.head(5)
X.head(5)
y.head(5)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 3512, shuffle=1)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
modelim = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
df_comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df_comp
from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(y_test, y_pred)

MSE
from sklearn.metrics import mean_absolute_error



MSA = mean_absolute_error(y_test, y_pred)

MSA
import math



RMSE = math.sqrt(MSE)

RMSE
modelim.score(X,y)
import statsmodels.api as stat

stmodel = stat.OLS(y, X).fit()

stmodel.summary()