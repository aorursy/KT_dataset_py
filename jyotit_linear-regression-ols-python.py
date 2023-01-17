# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv") #Training Data

df_test = pd.read_csv("../input/test.csv") #Testing Data

df_train.head()

df_train.info() #Gives an idea of any empty data, dropped it as x was an outlier value

df_train.dropna(inplace=True)
X_train = df_train["x"].values # X - a numpy array

Y_train = df_train["y"].values # Y - a numpy array

X_test = df_test["x"].values # X - a numpy array

Y_test = df_test["y"].values # Y - a numpy array
plt.scatter(X_train,Y_train)
m_X = X_train.mean()

m_Y = Y_train.mean()

s_X = X_train.std()

s_Y = Y_train.std()
r = np.corrcoef(X_train,Y_train)[1,0] #Correlation Coefficient, it is close to 1, hence strong positive relation

r
m = r*(s_Y/s_X) #Slope

b = m_Y - m * m_X  #Intercept
Y_pred = m * X_test + b #OLS Regression Equation
RMSE = np.sqrt(((Y_test - Y_pred) ** 2 ).mean()) #RMSE to check the performance of our model

RMSE
plt.scatter(X_test,Y_test)

plt.plot(X_test, Y_pred, 'r')
#Draw Residual plot to check if line a best fit - since no pattern, so good fit

plt.scatter(X_test,Y_test-Y_pred)