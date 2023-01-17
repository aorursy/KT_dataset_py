# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=1004)
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred[:10])
print('Coefficients are \n', regressor.coef_)
print('R2 Score is %.2f' % r2_score(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor

regressor_rf = RandomForestRegressor()
regressor_rf.fit(X_train, y_train)
y_pred = regressor_rf.predict(X_test)

print('R2 Score is %.2f' % r2_score(y_test, y_pred))
from sklearn.tree import DecisionTreeRegressor

regressor_dt = RandomForestRegressor()
regressor_dt.fit(X_train, y_train)
y_pred = regressor_dt.predict(X_test)

print('R2 Score is %.2f' % r2_score(y_test, y_pred))
import xgboost

regressor_xgb = xgboost.XGBRegressor()

regressor_xgb.fit(X_train,y_train)

y_pred = regressor_xgb.predict(X_test)

print('R2 Score is %.2f' % r2_score(y_test, y_pred))
import lightgbm as lgb

regressor_lgb = lgb.LGBMRegressor()

regressor_lgb.fit(X_train,y_train)

y_pred = regressor_lgb.predict(X_test)

print('R2 Score is %.2f' % r2_score(y_test, y_pred))
from sklearn.svm import SVR

regressor_svr = SVR()

regressor_svr.fit(X_train,y_train)

y_pred = regressor_svr.predict(X_test)

print('R2 Score is %.2f' % r2_score(y_test, y_pred))