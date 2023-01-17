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
dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 8].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
import math
from sklearn.metrics import mean_squared_error
rmse_non_optimized = math.sqrt(mean_squared_error(y_test,y_pred))
rmse_non_optimized
# backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((500,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,2,3,5,6,7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# not the best fit
X_opt = X[:, [0,1,2,5,6,7]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
regressor = LinearRegression()
regressor.fit(X_train[:, [0,1,2,4,5,6]], y_train)
y_pred_opt = regressor.predict(X_test[:, [0,1,2,4,5,6]])

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(y_test,y_pred_opt))
rmse