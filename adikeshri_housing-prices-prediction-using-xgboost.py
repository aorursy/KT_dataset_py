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


from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('../input/train.csv')

to_drop=['PoolQC','Fence','MiscFeature','Alley']

df.drop(to_drop,axis=1,inplace=True)

to_cat=[ 'SaleCondition','SaleType','PavedDrive','GarageCond','GarageQual','GarageFinish','GarageType','FireplaceQu','Functional','KitchenQual','Electrical','CentralAir','HeatingQC','Heating','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','MasVnrType','Foundation','ExterCond','ExterQual','Exterior2nd','Exterior1st','RoofMatl','RoofStyle','HouseStyle','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType']

#df[to_cat]=df[to_cat].astype('category')

to_str=['GarageCond','GarageQual','GarageFinish','GarageType','FireplaceQu','Electrical','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','MasVnrType']

df[to_str]=df[to_str].astype(str)

df[to_cat]=df[to_cat].apply(LabelEncoder().fit_transform)

#print(df['SaleCondition'].head())

#print(df['SaleCondition'].nunique())

df['LotFrontage'].fillna(df['LotFrontage'].median(),inplace=True)

df['MasVnrArea'].fillna(df['MasVnrArea'].median(),inplace=True)

df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(),inplace=True)

#print(df.info())

#from sklearn.linear_model import LinearRegression

import xgboost

import matplotlib.pyplot as plt

x_train=df.drop('SalePrice',axis=1)

y_train=df['SalePrice']

reg=xgboost.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)

#reg=LinearRegression()

reg.fit(x_train,y_train)

x_test=pd.read_csv('../input/test.csv')

x_test.drop(to_drop,axis=1,inplace=True)

new_to_str=to_str+['SaleType','Functional','KitchenQual','Exterior2nd','Exterior1st','MSZoning','Utilities']

x_test[new_to_str]=x_test[new_to_str].astype(str)

x_test[to_cat]=x_test[to_cat].apply(LabelEncoder().fit_transform)

x_test['LotFrontage'].fillna(x_test['LotFrontage'].median(),inplace=True)

x_test['MasVnrArea'].fillna(x_test['MasVnrArea'].median(),inplace=True)

x_test['BsmtFinSF1'].fillna(x_test['BsmtFinSF1'].median(),inplace=True)

x_test['BsmtFinSF2'].fillna(x_test['BsmtFinSF2'].median(),inplace=True)

x_test['BsmtUnfSF'].fillna(x_test['BsmtUnfSF'].median(),inplace=True)

x_test['TotalBsmtSF'].fillna(x_test['TotalBsmtSF'].median(),inplace=True)

x_test['BsmtFullBath'].fillna(x_test['BsmtFullBath'].median(),inplace=True)

x_test['BsmtHalfBath'].fillna(x_test['BsmtHalfBath'].median(),inplace=True)

x_test['GarageYrBlt'].fillna(x_test['GarageYrBlt'].median(),inplace=True)

x_test['GarageCars'].fillna(x_test['GarageCars'].median(),inplace=True)

x_test['GarageArea'].fillna(x_test['GarageArea'].median(),inplace=True)

#print(x_test.info())

y_test=reg.predict(x_test)

ans=pd.DataFrame({'Id':x_test['Id'],'SalePrice':y_test})

ans.to_csv('output.csv',index=False)