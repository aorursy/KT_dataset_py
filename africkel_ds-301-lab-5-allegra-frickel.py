import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df.head()
X=df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']]

y=df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

rnd_reg = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

rnd_reg.fit(X_train, y_train)

y_pred_rf = rnd_reg.predict(X_test)
print("RandomForestRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_rf)))
#ADABOOST

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor
ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=200, learning_rate=0.5)

ada_reg.fit(X_train, y_train)

y_pred_ada=ada_reg.predict(X_test)
print("AdaBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_ada)))
#GRADIENT BOOSTING

from xgboost import XGBRegressor

xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
print("XGBRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_xgb)))
#IMPROVING FEATURE SELECTION

X=df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF']]

y=df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
print("XGBRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_xgb)))
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#pull in as many features as possible to improve model accuracy

test_X=test[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF']]

test_y=test_X.values.reshape(-1,17)

predicted_price=xgb_reg.predict(test_X)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_price})

my_submission.to_csv('Lab5_FrickelAllegra.csv', index=False)