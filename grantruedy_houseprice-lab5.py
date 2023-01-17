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
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
X=df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']]

y=df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestRegressor
rnd_reg = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

rnd_reg.fit(X_train, y_train)

y_pred_rf = rnd_reg.predict(X_test)

print("RandomForestRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_rf)))
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=200, learning_rate=0.5)

ada_reg.fit(X_train, y_train)

y_pred_ada=ada_reg.predict(X_test)
print("AdaBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_ada)))
from xgboost import XGBRegressor
xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
print("XGBRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_xgb)))
X=df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF']]

y=df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
print("XGBRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_xgb)))
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_X=test[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF']]

test_y=test_X.values.reshape(-1,17)

predicted_price=xgb_reg.predict(test_X)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_price})

my_submission.to_csv('Grant_XGBoost.csv', index=False)