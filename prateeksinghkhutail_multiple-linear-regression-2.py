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
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

dataset
#now first we will try multiple linear regression to find the relationship between mbap (outcome variable) and sscp & degreep (predictor variables).

X=dataset.iloc[:,[2,7]].values

Y=dataset.iloc[:,12].values.reshape(-1,1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train)



#predicting the test result

y_pred=regressor.predict(X_test)



df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': y_pred.flatten()})

df

#to find the intercept value we can use intercept_ from sklearn library

regressor.intercept_
#Let’s check out the coefficients for the predictors:

regressor.coef_
from sklearn.metrics import r2_score

r2_score(Y_test, y_pred)


X=dataset.iloc[:,[4,7]].values#now X contain columns hscp and degreep

Y=dataset.iloc[:,12].values.reshape(-1,1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)



#fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,Y_train)



#predicting the test result

y_pred=regressor.predict(X_test)



df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': y_pred.flatten()})

df
#to find the intercept value we can use intercept_ from sklearn library

regressor.intercept_
#Let’s check out the coefficients for the predictors:

regressor.coef_
from sklearn.metrics import r2_score

r2_score(Y_test, y_pred)