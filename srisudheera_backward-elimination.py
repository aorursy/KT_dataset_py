import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/50_Startups.csv')
data.head(4)
data.State.value_counts()
data.isnull().sum()
data=pd.get_dummies(data)
data.head(4)
X=data.drop(['Profit'],axis=1)
X.head(4)
Y=data.Profit
Y.head(4)
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
from math import sqrt

mse=round((mean_squared_error(Y_test,Y_pred))/100, 2)
rmse = round((sqrt(mse))/100 ,2)
mse ,rmse
import statsmodels.api as sm
X=sm.add_constant(X)
model=sm.OLS(Y,X).fit()
model.summary()
X=X.drop(['Administration'],axis=1)
model=sm.OLS(Y,X).fit()
model.summary()
X=X.drop(['Marketing Spend'],axis=1)
model=sm.OLS(Y,X).fit()
model.summary()
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
mse=round((mean_squared_error(Y_test,Y_pred))/100, 2)
rmse = round((sqrt(mse))/100 ,2)
mse ,rmse