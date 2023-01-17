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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/bike_shares_r.csv")



df.shape
df.columns
df.head


df.duplicated().sum()
df = df.drop_duplicates()
df.duplicated().sum()
l_column = list(df.columns) # Making a list out of column names

len_feature = len(l_column) # Length of column vector list

l_column


X = df[l_column[0:len_feature-3]]

Y = df[l_column[len_feature-2]]
X.columns


print("Feature set size:",X.shape)

print("Variable set size:",Y.shape)
X.head()
Y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

print("Training feature set size X:",X_train.shape)

print("Test feature set size X:",X_test.shape)

print("Training variable set size Y:",y_train.shape)

print("Test variable set size Y:",y_test.shape)

from sklearn.linear_model import LinearRegression

from sklearn import metrics
lm = LinearRegression() # Creating a Linear Regression object 'lm'
lm.fit(X_train,y_train) 
train_pred = lm.predict(X_train)

test_pred = lm.predict(X_test)
train_pred
test_pred
print("The intercept term of the linear model:", lm.intercept_)
print("The coefficients of the linear model:", lm.coef_)


import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))

plt.title("Actual vs. predicted count",fontsize=25)

plt.xlabel("Actual test count",fontsize=18)

plt.ylabel("Predicted count", fontsize=18)

plt.scatter(x=y_test,y=test_pred)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))

plt.title("Actual vs. predicted count",fontsize=25)

plt.xlabel("Actual test count",fontsize=18)

plt.ylabel("Predicted count", fontsize=18)

plt.scatter(x=y_train,y=train_pred)
print("Mean absolute error (MAE):", metrics.mean_absolute_error(y_train,train_pred))

print("Mean square error (MSE):", metrics.mean_squared_error(y_train,train_pred))

print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(y_train,train_pred)))
print("Mean absolute error (MAE):", metrics.mean_absolute_error(y_test,test_pred))

print("Mean square error (MSE):", metrics.mean_squared_error(y_test,test_pred))

print("Root mean square error (RMSE):", np.sqrt(metrics.mean_squared_error(y_test,test_pred)))


def mean_absolute_percentage_error(y_train, train_pred): 

    y_train, train_pred = np.array(y_train), np.array(train_pred)

    return np.mean(np.abs((y_train - train_pred) / y_train)) * 100

mean_absolute_percentage_error(y_train, train_pred)