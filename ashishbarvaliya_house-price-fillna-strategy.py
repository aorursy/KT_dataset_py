# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading data

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.shape, test.shape
train.isna().sum().sort_values()[-19:-1]
test.isna().sum().sort_values()[-33:-1]
#Filling with "NA" string

for col in ['Alley','FireplaceQu','Fence','MiscFeature','PoolQC']:

    train[col].fillna('NA', inplace=True)

    test[col].fillna('NA', inplace=True)
train['LotFrontage'].value_counts()
train['LotFrontage'].fillna(train["LotFrontage"].value_counts().to_frame().index[0], inplace=True)

test['LotFrontage'].fillna(test["LotFrontage"].value_counts().to_frame().index[0], inplace=True)
train[['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']].isna().head(7)
for col in ['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']:

    train[col].fillna('NA',inplace=True)

    test[col].fillna('NA',inplace=True)
for col in ['BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2','BsmtExposure']:

    train[col].fillna('NA',inplace=True)

    test[col].fillna('NA',inplace=True)
train['Electrical'].value_counts()
train['Electrical'].fillna('SBrkr',inplace=True)
missings = ['GarageCars','GarageArea','KitchenQual','Exterior1st','SaleType','TotalBsmtSF','BsmtUnfSF','Exterior2nd',

            'BsmtFinSF1','BsmtFinSF2','BsmtFullBath','Functional','Utilities','BsmtHalfBath','MSZoning']

train[missings].head()
numerical=['GarageCars','GarageArea','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath']

categorical = ['KitchenQual','Exterior1st','SaleType','Exterior2nd','Functional','Utilities','MSZoning']
# using Imputer class of sklearn libs.

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median',axis=0)

imputer.fit(test[numerical] + train[numerical])

test[numerical] = imputer.transform(test[numerical])

train[numerical] = imputer.transform(train[numerical])
for i in categorical:

    train[i].fillna(train[i].value_counts().to_frame().index[0], inplace=True)

    test[i].fillna(test[i].value_counts().to_frame().index[0], inplace=True)    
train[train['MasVnrType'].isna()][['SalePrice','MasVnrType','MasVnrArea']]
print(train[train['MasVnrType']=='None']['SalePrice'].median())

print(train[train['MasVnrType']=='BrkFace']['SalePrice'].median())

print(train[train['MasVnrType']=='Stone']['SalePrice'].median())

print(train[train['MasVnrType']=='BrkCmn']['SalePrice'].median())
train['MasVnrArea'].fillna(181000,inplace=True)

test['MasVnrArea'].fillna(181000,inplace=True)



train['MasVnrType'].fillna('NA',inplace=True)

test['MasVnrType'].fillna('NA',inplace=True)
print(train.isna().sum().sort_values()[-5:-1])

print(test.isna().sum().sort_values()[-5:-1])
# train.to_csv('new_train',index=False)

# test.to_csv('new_test',index=False)