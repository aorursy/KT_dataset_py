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
import numpy as np

import pandas as pd

import os
#Data Importing

train_d=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_d=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train_d.head()
#EDA of response variable

import matplotlib.pyplot as plt

plt.hist(train_d['SalePrice'])

plt.ylabel('SalePrice')

plt.show()
train_d['SalePrice'].describe()
# Missing data in train

train_nas = train_d.isnull().sum()

train_nas = train_nas[train_nas>0]

train_nas.sort_values(ascending=False)
#missing data percent plot

total = train_d.isnull().sum().sort_values(ascending=False)

percent = (train_d.isnull().sum()/train_d.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# Handle missing values for features where median/mean or most common value doesn't make sense

# Alley : data description says NA means "no alley access"

train_d.loc[:, "Alley"] = train_d.loc[:, "Alley"].fillna("None")

# BedroomAbvGr : NA most likely means 0

train_d.loc[:, "BedroomAbvGr"] = train_d.loc[:, "BedroomAbvGr"].fillna(0)

# BsmtQual etc : data description says NA for basement features is "no basement"

train_d.loc[:, "BsmtQual"] = train_d.loc[:, "BsmtQual"].fillna("No")

train_d.loc[:, "BsmtCond"] = train_d.loc[:, "BsmtCond"].fillna("No")

train_d.loc[:, "BsmtExposure"] = train_d.loc[:, "BsmtExposure"].fillna("No")

train_d.loc[:, "BsmtFinType1"] = train_d.loc[:, "BsmtFinType1"].fillna("No")

train_d.loc[:, "BsmtFinType2"] = train_d.loc[:, "BsmtFinType2"].fillna("No")

train_d.loc[:, "BsmtFullBath"] = train_d.loc[:, "BsmtFullBath"].fillna(0)

train_d.loc[:, "BsmtHalfBath"] = train_d.loc[:, "BsmtHalfBath"].fillna(0)

train_d.loc[:, "BsmtUnfSF"] = train_d.loc[:, "BsmtUnfSF"].fillna(0)

# CentralAir : NA most likely means No

train_d.loc[:, "CentralAir"] = train_d.loc[:, "CentralAir"].fillna("N")

# Condition : NA most likely means Normal

train_d.loc[:, "Condition1"] = train_d.loc[:, "Condition1"].fillna("Norm")

train_d.loc[:, "Condition2"] = train_d.loc[:, "Condition2"].fillna("Norm")

# EnclosedPorch : NA most likely means no enclosed porch

train_d.loc[:, "EnclosedPorch"] = train_d.loc[:, "EnclosedPorch"].fillna(0)

# External stuff : NA most likely means average

train_d.loc[:, "ExterCond"] = train_d.loc[:, "ExterCond"].fillna("TA")

train_d.loc[:, "ExterQual"] = train_d.loc[:, "ExterQual"].fillna("TA")

# Fence : data description says NA means "no fence"

train_d.loc[:, "Fence"] = train_d.loc[:, "Fence"].fillna("No")

# FireplaceQu : data description says NA means "no fireplace"

train_d.loc[:, "FireplaceQu"] = train_d.loc[:, "FireplaceQu"].fillna("No")

train_d.loc[:, "Fireplaces"] = train_d.loc[:, "Fireplaces"].fillna(0)

# Functional : data description says NA means typical

train_d.loc[:, "Functional"] = train_d.loc[:, "Functional"].fillna("Typ")

# GarageType etc : data description says NA for garage features is "no garage"

train_d.loc[:, "GarageType"] = train_d.loc[:, "GarageType"].fillna("No")

train_d.loc[:, "GarageFinish"] = train_d.loc[:, "GarageFinish"].fillna("No")

train_d.loc[:, "GarageQual"] = train_d.loc[:, "GarageQual"].fillna("No")

train_d.loc[:, "GarageCond"] = train_d.loc[:, "GarageCond"].fillna("No")

train_d.loc[:, "GarageArea"] = train_d.loc[:, "GarageArea"].fillna(0)

train_d.loc[:, "GarageYrBlt"] = train_d.loc[:, "GarageYrBlt"].fillna(0)



train_d.loc[:, "GarageCars"] = train_d.loc[:, "GarageCars"].fillna(0)

# HalfBath : NA most likely means no half baths above grade

train_d.loc[:, "HalfBath"] = train_d.loc[:, "HalfBath"].fillna(0)

# HeatingQC : NA most likely means typical

train_d.loc[:, "HeatingQC"] = train_d.loc[:, "HeatingQC"].fillna("TA")

# KitchenAbvGr : NA most likely means 0

train_d.loc[:, "KitchenAbvGr"] = train_d.loc[:, "KitchenAbvGr"].fillna(0)

# KitchenQual : NA most likely means typical

train_d.loc[:, "KitchenQual"] = train_d.loc[:, "KitchenQual"].fillna("TA")

# LotFrontage : NA most likely means no lot frontage

train_d.loc[:, "LotFrontage"] = train_d.loc[:, "LotFrontage"].fillna(0)

# LotShape : NA most likely means regular

train_d.loc[:, "LotShape"] = train_d.loc[:, "LotShape"].fillna("Reg")

# MasVnrType : NA most likely means no veneer

train_d.loc[:, "MasVnrType"] = train_d.loc[:, "MasVnrType"].fillna("None")

train_d.loc[:, "MasVnrArea"] = train_d.loc[:, "MasVnrArea"].fillna(0)

# MiscFeature : data description says NA means "no misc feature"

train_d.loc[:, "MiscFeature"] = train_d.loc[:, "MiscFeature"].fillna("No")

train_d.loc[:, "MiscVal"] = train_d.loc[:, "MiscVal"].fillna(0)

# OpenPorchSF : NA most likely means no open porch

train_d.loc[:, "OpenPorchSF"] = train_d.loc[:, "OpenPorchSF"].fillna(0)

# PavedDrive : NA most likely means not paved

train_d.loc[:, "PavedDrive"] = train_d.loc[:, "PavedDrive"].fillna("N")

# PoolQC : data description says NA means "no pool"

train_d.loc[:, "PoolQC"] = train_d.loc[:, "PoolQC"].fillna("No")

train_d.loc[:, "PoolArea"] = train_d.loc[:, "PoolArea"].fillna(0)

# SaleCondition : NA most likely means normal sale

train_d.loc[:, "SaleCondition"] = train_d.loc[:, "SaleCondition"].fillna("Normal")

# ScreenPorch : NA most likely means no screen porch

train_d.loc[:, "ScreenPorch"] = train_d.loc[:, "ScreenPorch"].fillna(0)

# TotRmsAbvGrd : NA most likely means 0

train_d.loc[:, "TotRmsAbvGrd"] = train_d.loc[:, "TotRmsAbvGrd"].fillna(0)

# Utilities : NA most likely means all public utilities

train_d.loc[:, "Utilities"] = train_d.loc[:, "Utilities"].fillna("AllPub")

# WoodDeckSF : NA most likely means no wood deck

train_d.loc[:, "WoodDeckSF"] = train_d.loc[:, "WoodDeckSF"].fillna(0)
#missing data

total = train_d.isnull().sum().sort_values(ascending=False)

percent = (train_d.isnull().sum()/train_d.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#dealing with the last NAN

# train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

train_d = train_d.drop(train_d.loc[train_d['Electrical'].isnull()].index)

train_d.isnull().sum().max() #just checking that there's no missing data missing.
#Creating dummy variables from categorical variables

cst_vars=train_d.select_dtypes(['object']).columns   # identifying Categorical Variables

cst_vars

for col in cst_vars:

    dummy=pd.get_dummies(train_d[col],drop_first=True,prefix=col) # Creating dummy variables

    train_d=pd.concat([train_d,dummy],axis=1) # concating

    del train_d[col] #deleting old column

    print(col)

del dummy
# Transformation of response variable

train_d['log_SalePrice']=np.log(train_d['SalePrice'])

train_d['sqrt_SalePrice']=np.sqrt(train_d['SalePrice'])
plt.hist(train_d['log_SalePrice'])

plt.ylabel('log_SalePrice')

plt.show()
plt.hist(train_d['sqrt_SalePrice'])

plt.ylabel('sqrt_SalePrice')

plt.show()
# train test split

from sklearn.model_selection import train_test_split

train_1,train_2=train_test_split(train_d,test_size=0.3,random_state=2721)

train_1['log_SalePrice'].value_counts()

train_2['log_SalePrice'].value_counts()

x_train_1=train_1.drop(['SalePrice','log_SalePrice','sqrt_SalePrice','Id'],axis=1)

y_train_1=train_1['log_SalePrice']

x_train_1.shape
y_train_1.shape
x_train_2=train_2.drop(['SalePrice','log_SalePrice','sqrt_SalePrice','Id'],axis=1)

y_train_2=train_2['log_SalePrice']

x_train_2.shape
y_train_2.shape
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 

regressor.fit(x_train_1, y_train_1) 
y_train_2_pred_RF = regressor.predict(x_train_2)

from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train_2, y_train_2_pred_RF)))
# Test data Prediction

# Missing data in test

test_nas = test_d.isnull().sum()

test_nas = test_nas[test_nas>0]

test_nas.sort_values(ascending=False)
#missing data percent plot

total = test_d.isnull().sum().sort_values(ascending=False)

percent = (test_d.isnull().sum()/test_d.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# Handle missing values for features where median/mean or most common value doesn't make sense

# Alley : data description says NA means "no alley access"

test_d.loc[:, "Alley"] = test_d.loc[:, "Alley"].fillna("None")

# BedroomAbvGr : NA most likely means 0

test_d.loc[:, "BedroomAbvGr"] = test_d.loc[:, "BedroomAbvGr"].fillna(0)

# BsmtQual etc : data description says NA for basement features is "no basement"

test_d.loc[:, "BsmtQual"] = test_d.loc[:, "BsmtQual"].fillna("No")

test_d.loc[:, "BsmtCond"] = test_d.loc[:, "BsmtCond"].fillna("No")

test_d.loc[:, "BsmtExposure"] = test_d.loc[:, "BsmtExposure"].fillna("No")

test_d.loc[:, "BsmtFinType1"] = test_d.loc[:, "BsmtFinType1"].fillna("No")

test_d.loc[:, "BsmtFinType2"] = test_d.loc[:, "BsmtFinType2"].fillna("No")

test_d.loc[:, "BsmtFullBath"] = test_d.loc[:, "BsmtFullBath"].fillna(0)

test_d.loc[:, "BsmtHalfBath"] = test_d.loc[:, "BsmtHalfBath"].fillna(0)

test_d.loc[:, "BsmtUnfSF"] = test_d.loc[:, "BsmtUnfSF"].fillna(0)

# CentralAir : NA most likely means No

test_d.loc[:, "CentralAir"] = test_d.loc[:, "CentralAir"].fillna("N")

# Condition : NA most likely means Normal

test_d.loc[:, "Condition1"] = test_d.loc[:, "Condition1"].fillna("Norm")

test_d.loc[:, "Condition2"] = test_d.loc[:, "Condition2"].fillna("Norm")

# EnclosedPorch : NA most likely means no enclosed porch

test_d.loc[:, "EnclosedPorch"] = test_d.loc[:, "EnclosedPorch"].fillna(0)

# External stuff : NA most likely means average

test_d.loc[:, "ExterCond"] = test_d.loc[:, "ExterCond"].fillna("TA")

test_d.loc[:, "ExterQual"] = test_d.loc[:, "ExterQual"].fillna("TA")

# Fence : data description says NA means "no fence"

test_d.loc[:, "Fence"] = test_d.loc[:, "Fence"].fillna("No")

# FireplaceQu : data description says NA means "no fireplace"

test_d.loc[:, "FireplaceQu"] = test_d.loc[:, "FireplaceQu"].fillna("No")

test_d.loc[:, "Fireplaces"] = test_d.loc[:, "Fireplaces"].fillna(0)

# Functional : data description says NA means typical

test_d.loc[:, "Functional"] = test_d.loc[:, "Functional"].fillna("Typ")

# GarageType etc : data description says NA for garage features is "no garage"

test_d.loc[:, "GarageType"] = test_d.loc[:, "GarageType"].fillna("No")

test_d.loc[:, "GarageFinish"] = test_d.loc[:, "GarageFinish"].fillna("No")

test_d.loc[:, "GarageQual"] = test_d.loc[:, "GarageQual"].fillna("No")

test_d.loc[:, "GarageCond"] = test_d.loc[:, "GarageCond"].fillna("No")

test_d.loc[:, "GarageArea"] = test_d.loc[:, "GarageArea"].fillna(0)

test_d.loc[:, "GarageYrBlt"] = test_d.loc[:, "GarageYrBlt"].fillna(0)



test_d.loc[:, "GarageCars"] = test_d.loc[:, "GarageCars"].fillna(0)

# HalfBath : NA most likely means no half baths above grade

test_d.loc[:, "HalfBath"] = test_d.loc[:, "HalfBath"].fillna(0)

# HeatingQC : NA most likely means typical

test_d.loc[:, "HeatingQC"] = test_d.loc[:, "HeatingQC"].fillna("TA")

# KitchenAbvGr : NA most likely means 0

test_d.loc[:, "KitchenAbvGr"] = test_d.loc[:, "KitchenAbvGr"].fillna(0)

# KitchenQual : NA most likely means typical

test_d.loc[:, "KitchenQual"] = test_d.loc[:, "KitchenQual"].fillna("TA")

# LotFrontage : NA most likely means no lot frontage

test_d.loc[:, "LotFrontage"] = test_d.loc[:, "LotFrontage"].fillna(0)

# LotShape : NA most likely means regular

test_d.loc[:, "LotShape"] = test_d.loc[:, "LotShape"].fillna("Reg")

# MasVnrType : NA most likely means no veneer

test_d.loc[:, "MasVnrType"] = test_d.loc[:, "MasVnrType"].fillna("None")

test_d.loc[:, "MasVnrArea"] = test_d.loc[:, "MasVnrArea"].fillna(0)

# MiscFeature : data description says NA means "no misc feature"

test_d.loc[:, "MiscFeature"] = test_d.loc[:, "MiscFeature"].fillna("No")

test_d.loc[:, "MiscVal"] = test_d.loc[:, "MiscVal"].fillna(0)

# OpenPorchSF : NA most likely means no open porch

test_d.loc[:, "OpenPorchSF"] = test_d.loc[:, "OpenPorchSF"].fillna(0)

# PavedDrive : NA most likely means not paved

test_d.loc[:, "PavedDrive"] = test_d.loc[:, "PavedDrive"].fillna("N")

# PoolQC : data description says NA means "no pool"

test_d.loc[:, "PoolQC"] = test_d.loc[:, "PoolQC"].fillna("No")

test_d.loc[:, "PoolArea"] = test_d.loc[:, "PoolArea"].fillna(0)

# SaleCondition : NA most likely means normal sale

test_d.loc[:, "SaleCondition"] = test_d.loc[:, "SaleCondition"].fillna("Normal")

# ScreenPorch : NA most likely means no screen porch

test_d.loc[:, "ScreenPorch"] = test_d.loc[:, "ScreenPorch"].fillna(0)

# TotRmsAbvGrd : NA most likely means 0

test_d.loc[:, "TotRmsAbvGrd"] = test_d.loc[:, "TotRmsAbvGrd"].fillna(0)

# Utilities : NA most likely means all public utilities

test_d.loc[:, "Utilities"] = test_d.loc[:, "Utilities"].fillna("AllPub")

# WoodDeckSF : NA most likely means no wood deck

test_d.loc[:, "WoodDeckSF"] = test_d.loc[:, "WoodDeckSF"].fillna(0)
#missing data

total = test_d.isnull().sum().sort_values(ascending=False)

percent = (test_d.isnull().sum()/test_d.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#dealing with the last NAN

# train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

test_d = test_d.drop(test_d.loc[test_d['MSZoning'].isnull()].index)

test_d.isnull().sum().max() #just checking that there's no missing data missing.
#missing data

total = test_d.isnull().sum().sort_values(ascending=False)

percent = (test_d.isnull().sum()/test_d.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#dealing with the last NAN

# train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

test_d = test_d.drop(test_d.loc[test_d['TotalBsmtSF'].isnull()].index)

test_d.isnull().sum().max() #just checking that there's no missing data missing.
#missing data

total = test_d.isnull().sum().sort_values(ascending=False)

percent = (test_d.isnull().sum()/test_d.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#dealing with the last NAN

# train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

test_d = test_d.drop(test_d.loc[test_d['SaleType'].isnull()].index)

test_d.isnull().sum().max() #just checking that there's no missing data missing.
#missing data

total = test_d.isnull().sum().sort_values(ascending=False)

percent = (test_d.isnull().sum()/test_d.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#dealing with the last NAN

# train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

test_d = test_d.drop(test_d.loc[test_d['Exterior1st'].isnull()].index)

test_d.isnull().sum().max() #just checking that there's no missing data missing.
#missing data

total = test_d.isnull().sum().sort_values(ascending=False)

percent = (test_d.isnull().sum()/test_d.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
#Creating dummy variables from categorical variables

cst_vars=test_d.select_dtypes(['object']).columns   # identifying Categorical Variables

cst_vars

for col in cst_vars:

    dummy=pd.get_dummies(test_d[col],drop_first=True,prefix=col) # Creating dummy variables

    test_d=pd.concat([test_d,dummy],axis=1) # concating

    del test_d[col] #deleting old column

    print(col)

del dummy
test_d_n=test_d.drop('Id',axis=1)

list_col=train_d.columns

list_col_test=test_d_n.columns

x_train_1n=x_train_1[list_col_test]

x_train_2n=x_train_2[list_col_test]

x_train_1n.shape
test_d.shape
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 

regressor.fit(x_train_1n, y_train_1) 
y_train_2_pred_RF = regressor.predict(x_train_2n)

from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train_2, y_train_2_pred_RF)))
y_test_pred = regressor.predict(test_d_n)

y_test_pred_orig=np.exp(y_test_pred)
out={'ID':test_d['Id'],'SalesPrice':y_test_pred_orig}
df=pd.DataFrame(out)
df
df.to_csv("sample_submission_RF.csv", encoding='utf-8')