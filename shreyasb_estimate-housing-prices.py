import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

sns.set()

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train = pd.concat([train, test],axis=0)

print(train.describe())
null_cols = train.columns[train.isnull().any()]

quantity_null_col = train.isnull().any().sum()

quantity_per_col  = train[null_cols].isnull().sum()



print(quantity_per_col)

print('')

print("number of features with null columns")

print(quantity_null_col)
#---------------

#LotFrontage Imputation

imputer = SimpleImputer()

imputer_mode = SimpleImputer(strategy='most_frequent')



#imputer = SimpleImputer(Strategy = 'median')

imputed_data = imputer.fit_transform(train['LotFrontage'].values.reshape(2919, 1))

train['LotFrontage'] = imputed_data



#Alley

# train['Alley'] = train['Alley'].replace({'NA':1, 'Pave':2, 'Grvl':3})

train['Alley'].describe()

train['Alley'] = train['Alley'].fillna(1)

train['Alley'] = train['Alley'].replace({'Grvl':2, 'Pave':3})



#MasVnrType

# lb = LabelBinarizer()

imputed_data = imputer_mode.fit_transform(train['MasVnrType'].values.reshape(2919,1))

train['MasVnrType'] = imputed_data



#-------Street-----------

train['Street'] = train['Street'].replace({'Grvl':1, 'Pave':0})



#---------LotShape---------

train['LotShape'] = train['LotShape'].replace({'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1})



#------LandContour

train['LandContour'] = train['LandContour'].replace({'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1})



#----------Utilities---------

train['Utilities'] = train['Utilities'].fillna('AllPub')

train['Utilities'] = train['Utilities'].replace({'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1})



#---------LandSlope

train['LandSlope'] = train['LandSlope'].replace({'Gtl':3, 'Mod':2, 'Sev':1})



#------MasVnrArea---

train['MasVnrArea'] = imputer.fit_transform(train['MasVnrArea'].values.reshape(2919,1))



#------------Exterqual-----------

train['ExterQual'] = train['ExterQual'].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})



#------ExternCond-------

train['ExterCond'] = train['ExterCond'].replace({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1})



#-----BsmtQual------

train['BsmtQual'] = train['BsmtQual'].fillna(1)

train['BsmtQual'] = train['BsmtQual'].replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2})



#-----BsmtCond

train['BsmtCond'] = train['BsmtCond'].fillna(1)

train['BsmtCond'] = train['BsmtCond'].replace({'Ex':6, 'Gd':5, 'TA':4, 'Fa':3, 'Po':2})



#--------BsmtExposure

train['BsmtExposure'] = train['BsmtExposure'].fillna(1)

train['BsmtExposure'] = train['BsmtExposure'].replace({ 'Gd':5, 'Av':4, 'Mn':3, 'No':2})



#BsmtFinType1

train['BsmtFinType1'] = train['BsmtFinType1'].fillna(1)

train['BsmtFinType1'] = train['BsmtFinType1'].replace({'Unf':2, 'LwQ':3, 'Rec':4, 'BLQ':5, 'ALQ':6, 'GLQ':7})



#BsmtFinType2

train['BsmtFinType2'] = train['BsmtFinType2'].fillna(1)

train['BsmtFinType2'] = train['BsmtFinType2'].replace({'Unf':2, 'LwQ':3, 'Rec':4, 'BLQ':5, 'ALQ':6, 'GLQ':7})



#Heating

train['HeatingQC'] = train['HeatingQC'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})



#CentralAir

train['CentralAir'] = train['CentralAir'].replace({'Y':1, 'N':0})



#Electrical

train['Electrical'] = train['Electrical'].fillna(2)

train['Electrical'] = train['Electrical'].replace({'Mix':1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr':5})



#kitchenQual

train['KitchenQual'] = train['KitchenQual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})



#FireplceQu

train['FireplaceQu'] = train['FireplaceQu'].fillna(1)

train['FireplaceQu'] = train['FireplaceQu'].replace({'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})



#GarageType

imputed_data_GYB = imputer_mode.fit_transform(train['GarageType'].values.reshape(2919,1))

train['GarageType'] = imputed_data_GYB



#GarageYrBlt

imputed_data_GYB = imputer_mode.fit_transform(train['GarageYrBlt'].values.reshape(2919,1))

train['GarageYrBlt'] = imputed_data_GYB



#GarageFinish

train['GarageFinish'] = train['GarageFinish'].fillna(1)

train['GarageFinish'] = train['GarageFinish'].replace({'Unf':2, 'RFn':3, 'Fin':4})



#GarageQual

train['GarageQual'] = train['GarageQual'].fillna(1)

train['GarageQual'] = train['GarageQual'].replace({'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})



#GarageCond

train['GarageCond'] = train['GarageCond'].fillna(1)

train['GarageCond'] = train['GarageCond'].replace({'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})



#PavedDrive

train['PavedDrive'] = train['PavedDrive'].replace({'N':1, 'P':2, 'Y':3})



#PoolQC

train['PoolQC'] = train['PoolQC'].fillna(1)

train['PoolQC'] = train['PoolQC'].replace({'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})



#Fence

train['Fence'] = train['Fence'].fillna('None')



#BsmtFinSF1         1

imputed_data = imputer.fit_transform(train['BsmtFinSF1'].values.reshape(2919, 1))

train['BsmtFinSF1'] = imputed_data

# BsmtFinSF2         1

train['BsmtFinSF2'] = train['BsmtFinSF2'].fillna(0)

# BsmtFullBath       2

imputed_data = imputer_mode.fit_transform(train['BsmtFullBath'].values.reshape(2919,1))

train['BsmtFullBath'] = imputed_data

# BsmtHalfBath       2

imputed_data = imputer_mode.fit_transform(train['BsmtHalfBath'].values.reshape(2919,1))

train['BsmtHalfBath'] = imputed_data

# BsmtUnfSF          1

imputed_data = imputer.fit_transform(train['BsmtUnfSF'].values.reshape(2919, 1))

train['BsmtUnfSF'] = imputed_data

# GarageArea         1

imputed_data = imputer.fit_transform(train['GarageArea'].values.reshape(2919, 1))

train['GarageArea'] = imputed_data

# GarageCars         1

imputed_data = imputer_mode.fit_transform(train['GarageCars'].values.reshape(2919,1))

train['GarageCars'] = imputed_data

# KitchenQual        1

imputed_data = imputer_mode.fit_transform(train['KitchenQual'].values.reshape(2919,1))

train['KitchenQual'] = imputed_data

# TotalBsmtSF        1

imputed_data = imputer.fit_transform(train['TotalBsmtSF'].values.reshape(2919, 1))

train['TotalBsmtSF'] = imputed_data

# Utilities          2

lis_one_hot = ['MSZoning', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',

                'BldgType', 'HouseStyle', 'RoofMatl','RoofStyle', 'Exterior1st', 'Exterior2nd',

                'MasVnrType', 'Foundation', 'Heating', 'Functional', 'GarageType', 'Fence',

                'MiscFeature', 'SaleType', 'SaleCondition']



one_hot = pd.get_dummies(train, columns=lis_one_hot)
Ids = one_hot['Id']

temp_train = one_hot.iloc[:1460]

temp_test = one_hot.iloc[1459:]

# id_train = one_hot['Id'].iloc[:1461]

id_test = one_hot['Id'].iloc[1460:]

# prnt(id_test)

# print(temp_train)
Y = temp_train['SalePrice']

temp_train = temp_train.drop(['Id', 'SalePrice'], axis=1)

train_x, val_x, train_y, val_y = train_test_split(temp_train, Y, random_state = 42)

print(train_x.shape, val_x.shape, train_y.shape, val_y.shape)
number_trees = []

mse_train = []

mse_val = []

for i in range(100,1100,100):

    print("starting training for: ", i)

    number_trees.append(i)

    rr = RandomForestRegressor(n_estimators=i, max_depth=25,max_features=60)

    rr.fit(train_x, train_y)

    print("---predicting---")

    predict_train = rr.predict(train_x)

    predict_val = rr.predict(val_x)

    print("---mse---")

    mse_train.append(mean_squared_error(train_y, predict_train))

    mse_val.append(mean_squared_error(val_y, predict_val))

    

fig = plt.figure()

ax = plt.axes()

plt.plot(number_trees, mse_train,color='r')

plt.plot(number_trees, mse_val, color='g')

plt.show()
number_trees = []

mse_train = []

mse_val = []



for i in range(2,100,5):

    print("starting training for: ", i)

    number_trees.append(i)

    rr = RandomForestRegressor(n_estimators=200, max_depth=25,max_features=i)

    rr.fit(train_x, train_y)

    print("---predicting---")

    predict_train = rr.predict(train_x)

    predict_val = rr.predict(val_x)

    print("---mse---")

    mse_train.append(mean_squared_error(train_y, predict_train))

    mse_val.append(mean_squared_error(val_y, predict_val))

    

fig = plt.figure()

ax = plt.axes()

plt.plot(number_trees, mse_train,color='r')

plt.plot(number_trees, mse_val, color='g')

plt.show()
number_trees = []

mse_train = []

mse_val = []



for i in range(2,100,10):

    print("starting training for: ", i)

    number_trees.append(i)

    rr = RandomForestRegressor(n_estimators=200, max_depth=i,max_features=60)

    rr.fit(train_x, train_y)

    print("---predicting---")

    predict_train = rr.predict(train_x)

    predict_val = rr.predict(val_x)

    print("---mse---")

    mse_train.append(mean_squared_error(train_y, predict_train))

    mse_val.append(mean_squared_error(val_y, predict_val))

    

fig = plt.figure()

ax = plt.axes()

plt.plot(number_trees, mse_train,color='r')

plt.plot(number_trees, mse_val, color='g')

plt.show()
model = RandomForestRegressor(n_estimators=200, max_depth = 20, max_features=60)

model.fit(temp_train, Y)
temp_test = temp_test.drop(['SalePrice', 'Id'], axis=1)

submission = model.predict(temp_test)