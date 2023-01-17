import numpy as np

import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
#to see matplolib graphs in notebook

%matplotlib inline  
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print (train.shape,test.shape)
train.head()
test.head()
NAs = pd.concat([train.isnull().sum(),test.isnull().sum()],axis = 1, keys =['train','test'])

NAs[NAs.sum(axis = 1) > 1]
train = train.drop(['LotFrontage','Alley','FireplaceQu','GarageType','GarageYrBlt','BsmtQual','BsmtCond',

                   'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'],axis = 1)

test = test.drop(['LotFrontage','Alley','FireplaceQu','GarageType','GarageYrBlt','BsmtQual','BsmtCond',

                   'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'],axis = 1)
NAs = pd.concat([train.isnull().sum(),test.isnull().sum()],axis = 1,keys = ['train','test'])

NAs[NAs.sum(axis = 1) > 1]

#NAs.info()
sns.barplot(x = 'Electrical',y = 'SalePrice',data = train)
fig , (axis1) = plt.subplots(1,figsize = (10,5))

sns.countplot(x = 'Electrical',ax = axis1, data = train)
train = train.drop(['MasVnrType','MasVnrArea','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical'],axis = 1)

test = test.drop(['MasVnrType','MasVnrArea','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical'],axis = 1)
NAs = pd.concat([train.isnull().sum(),test.isnull().sum()],axis = 1,keys = ['train','test'])

NAs[NAs.sum(axis = 1) > 1]
cormat = train.corr()

fig , ax = plt.subplots(figsize = (12,12))

sns.heatmap(cormat , vmax = 0.8 , square = True)
train.columns
test.info()
train = train.drop(['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape',

       'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',

       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 

       'OverallCond', 'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating',

       'HeatingQC', 'CentralAir', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces','PavedDrive',

       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition'],axis = 1)

test = test.drop(['MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape',

       'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',

       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 

       'OverallCond', 'RoofStyle', 'RoofMatl','Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating',

       'HeatingQC', 'CentralAir', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces','PavedDrive',

       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition'],axis = 1)
train.head()
test.head()
cr = train.corr()

fig , ax = plt.subplots(figsize = (12,12))

sns.heatmap(cr , vmax = .8 , square = True , annot = True)
train = train.drop(['YearBuilt', 'YearRemodAdd', 'GarageCars', 'FullBath'],axis = 1)

test = test.drop(['YearBuilt', 'YearRemodAdd', 'GarageCars', 'FullBath'],axis = 1)
train.info()
test.info()
test['GarageArea'].fillna(test['GarageArea'].median(),inplace = 1)

test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median(),inplace = 1)
#training

train = train.drop(["Id"],axis = 1)

#test = test.drop(["1stFlrSF"],axis = 1)

X_train = train.drop("SalePrice",axis = 1)

Y_train = train["SalePrice"]

X_test = test.drop("Id",axis = 1).copy()
random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)