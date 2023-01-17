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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from scipy import stats

from scipy.stats import norm, skew
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head(10)
train.describe()
corr = train.corr()

plt.figure(figsize=(12,8))

sns.heatmap(corr)

plt.show()
plt.figure(figsize = (5,20))

sns.heatmap(corr[["SalePrice"]].sort_values(by = "SalePrice", ascending = False).head(20),vmin = -1, annot = True)

plt.show()
imp_feat = corr.index[abs(corr["SalePrice"])>0.4]
plt.figure(figsize = (12,8))

sns.heatmap(train[imp_feat].corr(), annot = True, cmap="RdYlGn")

plt.show()
columns = ["SalePrice","OverallQual","GrLivArea", "TotalBsmtSF", "YearBuilt", "GarageCars" ,"FullBath"]

sns.pairplot(train[columns],height = 2.5)

plt.show()
plt.figure(figsize = (12,8))

sns.barplot(train["OverallQual"],train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")

plt.title("OverallQualilty vs SalePrice")

plt.show()
plt.figure(figsize = (12,8))

sns.scatterplot(x = train["TotalBsmtSF"],y= train["SalePrice"])

plt.xlabel("TotalBsmtSF")

plt.ylabel("SalePrice")

plt.title("Basement area vs SalePrice")

plt.show()
plt.figure(figsize = (12,8))

sns.scatterplot(x = train["GrLivArea"],y= train["SalePrice"])

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.title("Ground Living Area vs SalePrice")

plt.show()
plt.figure(figsize = (16,8))

sns.boxplot(x = train["YearBuilt"],y= train["SalePrice"])

plt.xlabel("Year Built")

plt.xticks(rotation = 90)

plt.ylabel("SalePrice")

plt.title("Year Built vs SalePrice")

plt.show()
train_nan = train.isna().sum().sort_values(ascending = False)

train_nan[train_nan > 0]
train["PoolQC"].fillna("Gd", inplace = True)

train["MiscFeature"].fillna("Shed", inplace = True)

train["Alley"].fillna("Grvl", inplace = True)

train["Fence"].fillna("MnPrv", inplace = True)

train["FireplaceQu"].fillna("Gd", inplace = True)

train["LotFrontage"].fillna(69, inplace = True)

train['GarageCond'].fillna('No Garage', inplace=True)

train['GarageType'].fillna('No Garage', inplace=True)

train['GarageYrBlt'].fillna(round(train['GarageYrBlt'].median(), 1), inplace=True)

train['GarageFinish'].fillna('No Garage', inplace=True)

train['GarageQual'].fillna('No Garage', inplace=True)

train['BsmtExposure'].fillna('No Basement', inplace=True)

train['BsmtFinType2'].fillna('No Basement', inplace=True)

train['BsmtFinType1'].fillna('No Basement', inplace=True)

train['BsmtCond'].fillna('No Basement', inplace=True)

train['BsmtQual'].fillna('No Basement', inplace=True)

train['MasVnrArea'].fillna(0.0, inplace=True)

train['MasVnrType'].fillna('None', inplace=True)

train['Electrical'].fillna('Mixed', inplace=True)
train.isna().sum().sum()
train["SalePrice"].describe()
plt.figure(figsize = (8,8))

sns.boxplot(x= train["SalePrice"], data = train, orient = "v")

plt.show()
train = train.drop(train[train["SalePrice"] > 450000].index,axis = 0)
plt.figure(figsize = (8,8))

sns.boxplot(x= train["GrLivArea"], data = train, orient = "v")

plt.show()
train = train.drop(train[train["GrLivArea"] > 3500].index,axis = 0)
plt.figure(figsize = (8,8))

sns.boxplot(x= train["TotalBsmtSF"], data = train, orient = "v")

plt.show()
train = train.drop(train[train["TotalBsmtSF"] > 4000].index,axis = 0)
plt.figure(figsize = (8,8))

sns.boxplot(x= train["YearBuilt"], data = train, orient = "v")

plt.show()
train = train.drop(train[train["YearBuilt"] < 1880].index,axis = 0)
fig, ax = plt.subplots(1,2,figsize = (12,6))

sns.distplot(train["SalePrice"], fit = norm, ax = ax[0])



res = stats.probplot(train["SalePrice"],plot = plt)

plt.show()
train['SalePrice'] = np.log(train['SalePrice'])
fig, ax = plt.subplots(1,2,figsize = (12,6))

sns.distplot(train["SalePrice"], fit = norm, ax = ax[0])



res = stats.probplot(train["SalePrice"],plot = plt)

plt.show()
fig, ax = plt.subplots(1,2,figsize = (12,6))

sns.distplot(train["GrLivArea"], fit = norm, ax = ax[0])



res = stats.probplot(train["GrLivArea"],plot = plt)

plt.show()
train['GrLivArea'] = np.log(train['GrLivArea'])
fig, ax = plt.subplots(1,2,figsize = (12,6))

sns.distplot(train["GrLivArea"], fit = norm, ax = ax[0])



res = stats.probplot(train["GrLivArea"],plot = plt)

plt.show() 
fig, ax = plt.subplots(1,2,figsize = (12,6))

sns.distplot(train["TotalBsmtSF"], fit = norm, ax = ax[0])



res = stats.probplot(train["TotalBsmtSF"],plot = plt)

plt.show()
train.loc[train["TotalBsmtSF"] >0 , "TotalBsmtSF"] = np.log(train.loc[train["TotalBsmtSF"] >0 , "TotalBsmtSF"])
fig, ax = plt.subplots(1,2,figsize = (12,6))

sns.distplot(train[train["TotalBsmtSF"]>0]["TotalBsmtSF"], fit = norm, ax = ax[0])



res = stats.probplot(train[train["TotalBsmtSF"]>0]["TotalBsmtSF"],plot = plt)

plt.show()
train = pd.get_dummies(train)

train.head()
test_nan = test.isna().sum().sort_values(ascending = False)

test_nan[test_nan > 0]
test["PoolQC"].fillna("Gd", inplace = True)

test['GarageCond'].fillna('No Garage', inplace=True)

test['GarageType'].fillna('No Garage', inplace=True)

test['GarageYrBlt'].fillna(round(test['GarageYrBlt'].median(), 1), inplace=True)

test['GarageFinish'].fillna('No Garage', inplace=True)

test['GarageQual'].fillna('No Garage', inplace=True)

test['BsmtExposure'].fillna('No Basement', inplace=True)

test['BsmtFinType2'].fillna('No Basement', inplace=True)

test['BsmtFinType1'].fillna('No Basement', inplace=True)

test['BsmtCond'].fillna('No Basement', inplace=True)

test['BsmtQual'].fillna('No Basement', inplace=True)

test['MasVnrArea'].fillna(0.0, inplace=True)

test['MasVnrType'].fillna('None', inplace=True)

test['Electrical'].fillna('Mixed', inplace=True)

test["MiscFeature"].fillna("Shed", inplace = True)

test["Alley"].fillna("Grvl", inplace = True)

test["Fence"].fillna("MnPrv", inplace = True)

test["FireplaceQu"].fillna("Gd", inplace = True)

test["LotFrontage"].fillna(69, inplace = True)

test["MSZoning"].fillna("RL", inplace = True)

test["Functional"].fillna("Typ", inplace = True)

test["Utilities"].fillna("AllPub", inplace = True)

test["BsmtFullBath"].fillna(0, inplace = True)

test["BsmtHalfBath"].fillna(0, inplace = True)

test["TotalBsmtSF"].fillna(988, inplace = True)

test["SaleType"].fillna("WD", inplace = True)

test["GarageArea"].fillna(0, inplace = True)

test["Exterior1st"].fillna("VinylSd", inplace = True)

test["BsmtUnfSF"].fillna(0, inplace = True)

test["Exterior2nd"].fillna("VinylSd", inplace = True)

test["KitchenQual"].fillna("TA", inplace = True)

test["GarageCars"].fillna(2, inplace = True)

test["BsmtFinSF2"].fillna(0, inplace = True)

test["BsmtFinSF1"].fillna(0, inplace = True)
test = pd.get_dummies(test)

test.head()
from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from xgboost.sklearn import XGBRegressor
X = train.drop(["SalePrice"], axis = 1)

y = train["SalePrice"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 10)

print("Train set : ", X_train.shape,y_train.shape)

print("Test set : ", X_test.shape,y_test.shape)
forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X_train,y_train)
forest_pred = forest_model.predict(X_test)
print("Mean Squared error for Random Forest is : ",mean_squared_error(y_test,forest_pred))
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_test, y_test)], 

             verbose=False)
xboost_pred = my_model.predict(X_test)
print("Mean Squared error for XG Boost is : ",mean_squared_error(y_test,xboost_pred))
id_vals = test["Id"]
final_pred = my_model.predict(test.to_numpy(),validate_features=False)
final_output = pd.DataFrame()

final_output["Id"] = id_vals

final_output["SalePrice"] = final_pred
final_output.sort_values(by = "Id", ascending = True).head()
final_output.to_csv("final_submission.csv", index = False)