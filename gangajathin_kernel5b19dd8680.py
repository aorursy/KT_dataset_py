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
# Data Preprocessing Template



# Importing the dataset

dataset = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 11].values



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)



# Feature Scaling

"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

print(X.shape)
X = np.append(arr = np.ones((1599,1)).astype(int),values = X,axis = 1)

X_opt = X[:,[0,2,5,6,7,9,10,11]]

import statsmodels.api as sm

regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X_opt[:,1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 2)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)