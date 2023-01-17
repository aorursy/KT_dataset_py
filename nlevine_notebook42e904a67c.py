# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# train.corr()["SalePrice"]
predictors = ["OverallQual", "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch"]

train = pd.get_dummies(train)

for p in predictors:

    train[p] = train[p].fillna(train[p].mean())

print(train.columns)

rfr = RandomForestRegressor(n_estimators=100, min_samples_leaf=2)

rfr.fit(train[predictors], train["SalePrice"])

predictions_rfr = rfr.predict(train[predictors])

mse_rfr = mean_squared_error(train["SalePrice"], predictions_rfr)

rmse_rfr = mse_rfr**.5

print(rmse_rfr)
rfc = RandomForestClassifier(n_estimators=100, min_samples_leaf=2)

rfc.fit(train[predictors], train["SalePrice"])

predictions_rfc = rfc.predict(train[predictors])

mse_rfc = mean_squared_error(train["SalePrice"], predictions_rfc)

rmse_rfc = mse_rfc**.5

print(rmse_rfc)
lr = LinearRegression()

lr.fit(train[predictors], train["SalePrice"])

predictions_lr = lr.predict(train[predictors])

mse_lr = mean_squared_error(train["SalePrice"], predictions_lr)

rmse_lr = mse_lr**.5

print(rmse_lr)

