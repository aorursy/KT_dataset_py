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
#importing dataset
insurance = pd.read_csv("../input/insurance.csv")
print(insurance)
#Exploring Dataset
insurance.head()
print(insurance['children'].unique())
insurance.describe()
insurance.isnull()
insurance.info()
print(insurance['region'].unique())
insurance = insurance.drop(['region'], axis=1)
print(insurance)

#assignig X and Y

X = insurance.iloc[:, :5].values
y = insurance.iloc[:, -1].values
print(X)
print(y)
## Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, -1] = labelencoder.fit_transform(X[:, -1]).astype(int)
X[:, 1] = labelencoder.fit_transform(X[:, 1]).astype(int)
onehotencoder = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()


# Avoiding the Dummy Variable Trap
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
#Backward elimination to check for high significance feature


import statsmodels.formula.api as sm
a = 0
b = 0
a, b = X.shape
X = np.append(arr = np.ones((a, 1)).astype(int), values = X, axis = 1)
print (X.shape)

X_opt = X[:, [0, 1, 2, 4]]
regressorOLS = sm.OLS(endog = y, exog = X_opt).fit()
regressorOLS.summary()
# R squared Validation

from sklearn.metrics import r2_score
scr =  r2_score(y_test, y_pred)
