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
## Creating DataFrame using csv file
data=pd.read_csv("../input/hp_data.csv")
data.head()
from sklearn.linear_model import LinearRegression
## Storing sqft,yearsOld,floor,totalFloor,bhk values of all records in X variable
X=data.loc[:,['sqft','yearsOld','floor','totalFloor','bhk']]
## Storing Dependent Varibale price in Y variable
y=data.price
from sklearn.model_selection import train_test_split
## Splittiong Data into Train Data and Test Data into 70% and 30% respectively

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.3)
model=LinearRegression()
## Here we are training our linear Regression model with Trainig Data
model.fit(x_train,y_train)
## Here we are predicting price value using Test data
y_predict=model.predict(x_test)
y_predict[0:5]
y_test.head(5)
from sklearn.metrics import r2_score
## Here we are checking efficiency bettween predict value and test value

r2_score(y_test,y_predict)
