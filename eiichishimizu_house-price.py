import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import numpy.random as npRnd

import sys

import matplotlib

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import seaborn as sns



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_path = "../input/house-prices-advanced-regression-techniques/train.csv"

test_path = "../input/house-prices-advanced-regression-techniques/test.csv"

result_path = "submission.csv"



train = pd.read_csv(train_path)

test = pd.read_csv(test_path)
train.columns
corrmat = train.corr()



k = 10

cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index

cm = np.corrcoef(train[cols].values.T)



sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
plt.scatter(train["1stFlrSF"], train["SalePrice"])
train_run = train.drop(train[(train["1stFlrSF"] > 4000) & (train["SalePrice"] > 100000)].index)

plt.scatter(train_run["1stFlrSF"], train_run["SalePrice"])
plt.scatter(train_run["OverallQual"], train_run["SalePrice"])
train_run = train_run.drop(train_run[(train_run["OverallQual"] < 5) & (train_run["SalePrice"] > 200000)].index)

train_run = train_run.drop(train_run[(train_run["OverallQual"] < 10) & (train_run["SalePrice"] > 500000)].index)

plt.scatter(train_run["OverallQual"], train_run["SalePrice"])
plt.scatter(train_run["GrLivArea"], train_run["SalePrice"])
train_run = train_run.drop(train_run[(train_run["GrLivArea"] > 4000) & (train_run["SalePrice"] > 100000)].index)

plt.scatter(train_run["GrLivArea"], train_run["SalePrice"])
plt.scatter(train_run["GarageArea"], train_run["SalePrice"])
train_run = train_run.drop(train_run[(train_run["GarageArea"] > 1200) & (train_run["SalePrice"] > 70000)].index)

plt.scatter(train_run["GrLivArea"], train_run["SalePrice"])
from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.svm import SVR

from xgboost import XGBRegressor



linreg = LinearRegression()

lgbmr = LGBMRegressor()

gbr = GradientBoostingRegressor()

elastic = ElasticNetCV()

lasso = LassoCV()

ridge = RidgeCV()

svr = SVR()

xgbr = XGBRegressor()



models = [linreg, lgbmr, gbr, elastic, lasso, ridge, svr, xgbr,]



X = train_run[["OverallQual", "GrLivArea", "1stFlrSF"]].values

y = train_run["SalePrice"].values



for m in models:

  m.fit(X, y)
for m in models:

  print(m.score(X, y))
X_test = test[["OverallQual", "GrLivArea", "1stFlrSF"]].values



y_pred = models[1].predict(X_test)

print(y_pred)
test["SalePrice"] = y_pred

test[["Id", "SalePrice"]].to_csv(result_path, index=False)