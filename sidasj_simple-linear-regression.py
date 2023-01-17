# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/random-linear-regression/train.csv")

df_test = pd.read_csv("/kaggle/input/random-linear-regression/test.csv")

print("Train null values:",df_train.isna().sum())

print("Test null values:",df_test.isna().sum())

# As dataset contains null values but to fit linear regression we need to have no null values in our dataset

#### dropping the null values as they are of no use for us 

df_train.dropna(inplace=True)

df_test.dropna(inplace=True)



print("Train null values:",df_train.isna().sum())

print("Test null values:",df_test.isna().sum())
X_train = df_train["x"].to_frame()

y_train = df_train["y"].to_frame()#values.reshape(len(df_train["y"]),)

X_test = df_test["x"].to_frame()#values.reshape(len(df_test["x"]),)

y_test = df_test["y"].to_frame()#values.reshape(len(df_test["y"]),)

# Importing LinearRegression from linear model of sklearn

from sklearn.linear_model import LinearRegression

regressior = LinearRegression(fit_intercept=False,normalize=True,n_jobs=10000)

regressior.fit(X_train,y_train)
prediction = regressior.predict(X_test).reshape(len(X_test),)
from sklearn.metrics import r2_score



r2_score(y_test, prediction)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,prediction)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.figure(figsize=(20,8))

plt.scatter(X_test,y_test,label="Original")

plt.plot(X_test,prediction,label="Prediction",color="r")

plt.title("Linear Regression")

plt.xlabel("X values")

plt.ylabel("Y values")

plt.legend()

plt.show()