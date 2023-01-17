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
dataset = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
trainset= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
X= dataset.iloc[:,1:-1].values
y= dataset.iloc[:,-1].values
Xtest = trainset.iloc[:,1:]
print(Xtest)
from sklearn.impute import SimpleImputer
'''nan is a method under numpy which denotes the missing values in the array'''
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer2 = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer.fit(X)
X= imputer.transform(X)
X1= dataset.iloc[:,1:-1]
X1= pd.get_dummies(X1)
print(X1.head(10))
Xtest= pd.get_dummies(Xtest)
Xtest = Xtest.values
impute = SimpleImputer(missing_values=np.nan,strategy='mean')
impute.fit(Xtest)
Xtest = impute.transform(Xtest)
print(Xtest.shape)
X= X1.values
imputer.fit(X)
X= imputer.transform(X)
print(X.shape)
X = np.append(arr=np.ones((1460,1)).astype(int),values= X,axis=1)
print(X[:10,:10])
Xtest = np.append(arr=np.ones((1459,1)).astype(int),values= Xtest,axis=1)
import statsmodels.api as sm
'''Automatic Backward Elimination function'''
def BackwardElimination(x,sl):
  n= len(x[0])
  for i in range(n):
    regressor_ols= sm.OLS(endog=y,exog=x.astype('float64')).fit()
    pmax= max(regressor_ols.pvalues).astype(float)
    if(pmax > sl):
      for j in range(n-i):
        if(regressor_ols.pvalues[j] == pmax):          
          x = np.delete(x,j,1)
  print(regressor_ols.summary())
  return x
def BackwardElimination2(x,sl):
  n= len(x[0])
  for i in range(n):
    regressor_ols= sm.OLS(endog=y[:-1,],exog=x.astype('float64')).fit()
    pmax= max(regressor_ols.pvalues).astype(float)
    if(pmax > sl):
      for j in range(n-i):
        if(regressor_ols.pvalues[j] == pmax):          
          x = np.delete(x,j,1)
  print(regressor_ols.summary())
  return x
X_opt = X[:,[x for x in range(len(X[0]))]]
sl=0.04
X_red= BackwardElimination(X_opt,sl)
print(X_red)
Xopt = Xtest[:,[x for x in range(len(Xtest[0]))]]
sl=0.04
Xred= BackwardElimination2(Xopt,sl)
print(Xred)
print(X_red.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X_red,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred2= regressor.predict(Xtest)
print(y_test)
print(y_pred)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,y_pred)
print(mae)