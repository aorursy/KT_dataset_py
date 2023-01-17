import numpy as np

import pandas as pd

from sklearn.ensemble import  GradientBoostingRegressor
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train['totalSF'] =( train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']  )

train['total_bathrooms'] = (train['BsmtFullBath'] + 0.5*train['BsmtHalfBath'] + train['FullBath'] + 0.5*train['HalfBath'])

train['ageHouse'] = (train['YrSold'] - train['YearBuilt'] )

test['totalSF'] =( test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']  )

test['total_bathrooms'] = (test['BsmtFullBath'] + 0.5*test['BsmtHalfBath'] + test['FullBath'] + 0.5*test['HalfBath'])

test['ageHouse'] = (test['YrSold'] - test['YearBuilt'] )
train.drop(['Id','Utilities','PoolQC','MiscFeature','Alley','Fence','GarageYrBlt'] , axis=1 , inplace=True)

train.drop(['TotalBsmtSF' , '1stFlrSF' ,'2ndFlrSF'] , axis = 1 , inplace =True)

train.drop(['BsmtFullBath' , 'BsmtHalfBath' , 'FullBath' , 'HalfBath'] , axis=1 , inplace=True)

train = train.drop(columns = [ 'YearRemodAdd', 'GarageQual', 'GarageCond','YrSold','MoSold', 'YearBuilt', 'Heating'], axis = 1)

Y = train['SalePrice']

train = train.drop(columns = ['SalePrice'], axis = 1)

Id = test['Id']

test.drop(['Id','Utilities','PoolQC','MiscFeature','Alley','Fence','GarageYrBlt'] , axis=1 , inplace=True)

test.drop(['TotalBsmtSF' , '1stFlrSF' ,'2ndFlrSF'] , axis = 1 , inplace =True)

test.drop(['BsmtFullBath' , 'BsmtHalfBath' , 'FullBath' , 'HalfBath'] , axis=1 , inplace=True)

test = test.drop(columns = [ 'YearRemodAdd', 'GarageQual', 'GarageCond','YrSold','MoSold', 'YearBuilt', 'Heating'], axis = 1)
mode =  ['MasVnrArea' , 'Electrical' , 'MSZoning' , 'SaleType','Exterior1st','Exterior2nd','KitchenQual']

for col in mode:

    train[col]  = train[col].fillna(train[col].mode()[0])

for col in mode:

    test[col]  = test[col].fillna(test[col].mode()[0])

No = ['GarageType','GarageFinish',

                'BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual',

                'FireplaceQu','MasVnrType']

for col in No:

    train[col]=train[col].fillna('None')

for col in No:

    test[col]=test[col].fillna('None')

train['Functional'] = train['Functional'].fillna('Typ')

train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())

test['Functional'] = test['Functional'].fillna('Typ')

test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].median())

zero = ['total_bathrooms','totalSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','GarageArea','GarageCars' ]

for col in zero:

    train[col] = train[col].fillna(0)

for col in zero:

    test[col] = test[col].fillna(0)

train.head()
test.head()
train.dropna(inplace = True)

test.dropna(inplace = True)
Label = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Condition1', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterCond', 'Electrical', 'Functional', 'PavedDrive', 'SaleType', 'Exterior1st', 'HeatingQC', 'BsmtCond', 'Foundation', 'SaleCondition', 'CentralAir', 'ExterQual' ,'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Neighborhood', 'OverallQual', 'FireplaceQu', 'KitchenQual' , 'Condition2', 'Exterior2nd', 'GarageType', 'GarageFinish', 'BldgType', 'HouseStyle']

for feature in Label:

    ohe = pd.get_dummies(train[feature], prefix=feature)

    train = pd.concat([train, ohe], axis=1)

train = train.drop(columns = Label, axis = 1)

for feature in Label:

    ohe = pd.get_dummies(test[feature], prefix=feature)

    test = pd.concat([test, ohe], axis=1)

test = test.drop(columns = Label, axis = 1)
train = train.drop(columns = ['HouseStyle_2.5Fin','RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Membran', 'RoofMatl_ClyTile', 'Electrical_Mix', 'Exterior1st_Stone', 'Exterior1st_ImStucc', 'Condition2_RRNn', 'Condition2_RRAn', 'Condition2_RRAe', 'Exterior2nd_Other'], axis = 1)
train.head()
test.head()
model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

model.fit(X = train, y = Y)
test['SalePrice'] = model.predict(test)

test['Id'] = Id

test[['Id', 'SalePrice']].to_csv('price_submission.csv', index=False)