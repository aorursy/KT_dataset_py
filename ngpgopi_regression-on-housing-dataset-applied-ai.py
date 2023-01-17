import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.datasets import make_classification
train = pd.read_csv("../input/train.csv")

train.drop(['Id'], axis=1,inplace=True)

train.describe()
test = pd.read_csv("../input/test.csv")

Id = test['Id']

test.drop(['Id'], axis=1,inplace=True)

test.head()
missingValues = (train.isnull().sum())

temp = missingValues.where(missingValues > 0)

temp.dropna(inplace=True)

print(temp.sort_values(axis=0,ascending=False))
cNameswithGreaterthan600 = missingValues.where(missingValues > 600)

cNameswithGreaterthan600.dropna(inplace=True)

print(cNameswithGreaterthan600.sort_values(axis=0,ascending=False))

cNameswithGreaterthan600 = list(cNameswithGreaterthan600.index)

modifiedTrain = train.drop(cNameswithGreaterthan600,axis=1)

modifiedTest = test.drop(cNameswithGreaterthan600,axis=1)
missingValues = (modifiedTrain.isnull().sum())

temp = missingValues.where(missingValues > 0)

temp.dropna(inplace=True)

print(temp.sort_values(axis=0,ascending=False))
modifiedTrain['LotFrontage'].fillna(0,inplace=True) 

modifiedTest['LotFrontage'].fillna(0,inplace=True) 

modifiedTrain['GarageCond'].fillna('No',inplace=True) 

modifiedTest['GarageCond'].fillna('No',inplace=True) 

modifiedTrain.drop('GarageQual',inplace=True,axis=1) 

modifiedTest.drop('GarageQual',inplace=True,axis=1) 

modifiedTrain['GarageFinish'].fillna('No',inplace=True) 

modifiedTest['GarageFinish'].fillna('No',inplace=True) 

modifiedTrain['GarageYrBlt'].fillna(modifiedTrain['YearBuilt'],inplace=True)

modifiedTest['GarageYrBlt'].fillna(modifiedTest['YearBuilt'],inplace=True)

modifiedTrain['GarageType'].fillna('No',inplace=True) 

modifiedTest['GarageType'].fillna('No',inplace=True) 

modifiedTrain['BsmtFinType2'].fillna('No',inplace=True) 

modifiedTest['BsmtFinType2'].fillna('No',inplace=True) 

modifiedTrain['BsmtExposure'].fillna('No',inplace=True) 

modifiedTest['BsmtExposure'].fillna('No',inplace=True) 

modifiedTrain['BsmtFinType1'].fillna('No',inplace=True) 

modifiedTest['BsmtFinType1'].fillna('No',inplace=True) 

modifiedTrain.drop('BsmtQual',inplace=True,axis=1) 

modifiedTest.drop('BsmtQual',inplace=True,axis=1) 

modifiedTrain['BsmtCond'].fillna('No',inplace=True) 

modifiedTest['BsmtCond'].fillna('No',inplace=True) 

modifiedTrain['MasVnrArea'].fillna(0,inplace=True) 

modifiedTest['MasVnrArea'].fillna(0,inplace=True) 

modifiedTrain['MasVnrType'].fillna('No',inplace=True) 

modifiedTest['MasVnrType'].fillna('No',inplace=True) 

modifiedTrain['MasVnrType'].fillna('No',inplace=True) 

modifiedTest['MasVnrType'].fillna('No',inplace=True) 

modifiedTrain['Electrical'].fillna('Mix',inplace=True) 

modifiedTest['Electrical'].fillna('Mix',inplace=True) 
missingValues = (modifiedTrain.isnull().sum())

temp = missingValues.where(missingValues > 0)

temp.dropna(inplace=True)

print(temp.sort_values(axis=0,ascending=False))
missingValues = (modifiedTest.isnull().sum())

temp = missingValues.where(missingValues > 0)

temp.dropna(inplace=True)

print(temp.sort_values(axis=0,ascending=False))
modifiedTest['MSZoning'].fillna('RM',inplace=True) 

modifiedTest['Functional'].fillna('Mod',inplace=True) 

modifiedTest['BsmtHalfBath'].fillna(0,inplace=True) 

modifiedTest['BsmtFullBath'].fillna(0,inplace=True) 

modifiedTest['Utilities'].fillna('AllPub',inplace=True) 

modifiedTest['SaleType'].fillna('Oth',inplace=True) 

modifiedTest['GarageArea'].fillna(0,inplace=True) 

modifiedTest['GarageCars'].fillna(0,inplace=True) 

modifiedTest['KitchenQual'].fillna('TA',inplace=True) 

modifiedTest['TotalBsmtSF'].fillna(0,inplace=True) 

modifiedTest['BsmtUnfSF'].fillna(0,inplace=True) 

modifiedTest['BsmtFinSF2'].fillna(0,inplace=True) 

modifiedTest['BsmtFinSF1'].fillna(0,inplace=True) 

modifiedTest['Exterior2nd'].fillna('Other',inplace=True) 

modifiedTest['Exterior1st'].fillna('Other',inplace=True) 

missingValues = (modifiedTest.isnull().sum())

temp = missingValues.where(missingValues > 0)

temp.dropna(inplace=True)

print(temp.sort_values(axis=0,ascending=False))
grp = modifiedTrain.columns.to_series().groupby(modifiedTrain.dtypes).groups

for each in grp[np.dtype(object)]:

    modifiedTrain[each] = modifiedTrain[each].astype('category')

    modifiedTest[each] = modifiedTest[each].astype('category')
cat_columns = modifiedTrain.select_dtypes(['category']).columns

modifiedTrain[cat_columns] = modifiedTrain[cat_columns].apply(lambda x: x.cat.codes)

modifiedTest[cat_columns] = modifiedTest[cat_columns].apply(lambda x: x.cat.codes)
Xtrain = modifiedTrain.loc[:, modifiedTrain.columns != 'SalePrice']

Ytrain = modifiedTrain['SalePrice']

Xtest = modifiedTest
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 7, alpha = 10, n_estimators = 2000)

xg_reg.fit(Xtrain, Ytrain)
lasso = Lasso(alpha =0.0005, random_state=1)

lasso.fit(Xtrain,Ytrain)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

GBoost.fit(Xtrain,Ytrain)
YpredictedonTrain = xg_reg.predict(Xtrain)

Ypredicted2onTrain = GBoost.predict(Xtrain)

Ypredicted3onTrain = lasso.predict(Xtrain)

print(YpredictedonTrain.shape)

print(Ypredicted2onTrain.shape)

print(Ypredicted3onTrain.shape)
dfinal = pd.DataFrame({'a':YpredictedonTrain, 'b':Ypredicted2onTrain,'c':Ypredicted3onTrain})

dfinal.head()
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(dfinal,Ytrain)
a = xg_reg.predict(Xtest)

b = GBoost.predict(Xtest)

c = lasso.predict(Xtest)

dStack = pd.DataFrame({'a':a, 'b':b,'c':c})

dStack.head()
output = model_lgb.predict(dStack)
df = pd.DataFrame(output,index=Id,columns=['SalePrice'])

df.head()

df.to_csv('output2.csv')