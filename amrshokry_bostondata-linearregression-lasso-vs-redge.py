# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_boston

from sklearn.preprocessing import MaxAbsScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge
BostonData = load_boston()
#X Data

X = BostonData.data

print('X Data is \n' , X[:5])

print('X shape is ' , X.shape)

print('X Features are \n' , BostonData.feature_names)
#y Data

y = BostonData.target

print('y Data is \n' , y[:5])

print('y shape is ' , y.shape)
scaler = MaxAbsScaler(copy=True)

X = scaler.fit_transform(X)
print('X \n' , X[:5])

print('y \n' , y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
LassoRegressionModel = Lasso(alpha=0.01,random_state=33,normalize=False)

LassoRegressionModel.fit(X_train, y_train)
print('Lasso Regression Train Score is : ' , LassoRegressionModel.score(X_train, y_train))

print('Lasso Regression Test Score is : ' , LassoRegressionModel.score(X_test, y_test))

print('Lasso Regression Coef is : ' , LassoRegressionModel.coef_)

print('Lasso Regression intercept is : ' , LassoRegressionModel.intercept_)


y_pred = LassoRegressionModel.predict(X_test)

print('Predicted Value for Lasso Regression is : ' , y_pred[:10])
MAEValue = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error Value is : ', MAEValue)
RidgeRegressionModel = Ridge(alpha=0.01,random_state=33)

RidgeRegressionModel.fit(X_train, y_train)
print('Ridge Regression Train Score is : ' , RidgeRegressionModel.score(X_train, y_train))

print('Ridge Regression Test Score is : ' , RidgeRegressionModel.score(X_test, y_test))

print('Ridge Regression Coef is : ' , RidgeRegressionModel.coef_)

print('Ridge Regression intercept is : ' , RidgeRegressionModel.intercept_)
y_pred = RidgeRegressionModel.predict(X_test)

#print('Predicted Value for Ridge Regression is : ' , y_pred[:10])
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') 

print('Mean Absolute Error Value is : ', MAEValue)