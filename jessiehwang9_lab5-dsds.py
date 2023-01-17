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
train= pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test= pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
#drop columns has too many NA

train = train.drop(train[["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"]],axis=1)

test = test.drop(test[["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"]],axis=1)

train
import seaborn as sns

import matplotlib.pyplot as plt

plt.subplots(figsize=(20,20))

ax = plt.axes()

corr = train.corr()

sns.heatmap(corr)
#choose from features that has higher correlation from the heatmap

train1 = train[["SalePrice","OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","GarageCars","GarageArea"]]

test1 = test[["OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","GarageCars","GarageArea"]]

#add some might useful variables

train1["GarageFinish"] = train["GarageFinish"]

test1["GarageFinish"] = test["GarageFinish"]

train1["GarageType"] = train["GarageType"]

test1["GarageType"] = test["GarageType"]

train1["Foundation"] = train["Foundation"]

test1["Foundation"] = test["Foundation"]

train1["ExterQual"] = train["ExterQual"]

test1["ExterQual"] = test["ExterQual"]

train1["MasVnrType"] = train["MasVnrType"]

test1["MasVnrType"] = train["MasVnrType"]
plt.subplots(figsize=(20,20))

ax = plt.axes()

corr = train1.corr()

sns.heatmap(corr)
#from the heatmap, only ExterQual is a good feature

traindata = train1.drop(train[["MasVnrType","Foundation","GarageType","GarageFinish"]],axis=1)

testdata = test1.drop(train[["MasVnrType","Foundation","GarageType","GarageFinish"]],axis=1)

#fil Na

testdata["MasVnrArea"].fillna(method='ffill',inplace=True)

testdata.fillna(testdata.mean(),inplace=True)

traindata["MasVnrArea"].fillna(method='ffill',inplace=True)

traindata.fillna(testdata.mean(),inplace=True)
traindata["ExterQual"] = traindata["ExterQual"].map({"TA":1,"Gd":2,"Ex":3,"Fa":4})

testdata["ExterQual"] = testdata["ExterQual"].map({"TA":1,"Gd":2,"Ex":3,"Fa":4})

traindata.head()
X=traindata[["OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","GarageCars","GarageArea","ExterQual"]]

y=traindata[['SalePrice']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
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
X=train[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF']]

y=train[['SalePrice']]

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

my_submission.to_csv('submission_lab5.csv', index=False)