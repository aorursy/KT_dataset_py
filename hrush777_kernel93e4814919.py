# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

print(train_data.shape)

train_data.head()
sns.boxplot(train_data.GrLivArea)
train_data = train_data[train_data.GrLivArea < 4000]
train_data.shape
sns.scatterplot(train_data.LotArea,train_data.SalePrice)
sns.boxplot(train_data.LotArea)
train_data = train_data[train_data.LotArea <= 50000]

print(train_data.shape)
sns.scatterplot(train_data.TotalBsmtSF,train_data.SalePrice)
sns.boxplot(train_data.TotalBsmtSF)
train_data = train_data[train_data.TotalBsmtSF <= 2500]

print(train_data.shape)
sns.scatterplot(train_data.LotFrontage,train_data.SalePrice)
print(train_data.shape)
train_data.head()
test_data = pd.read_csv('../input/test.csv')

test_data.head()
_id = test_data['Id'].values
target = train_data['SalePrice'].values

train_data = train_data.drop('SalePrice',1)
print(train_data.shape)

print(test_data.shape)
train_data.columns
test_data.columns
total_data = pd.concat([train_data,test_data],0)
total_data.shape
total_data.head()
total_data.Id = total_data.Id.values.astype('int')
total_data.head()
# total_data.MSSubClass = total_data.MSSubClass.fillna(0)

# total_data.MSZoning = total_data.MSZoning.fillna('None')

# total_data.LotFrontage = total_data.LotFrontage.fillna(list(total_data.LotFrontage.median())[0])

# total_data.LotArea = total_data.LotArea.fillna(list(total_data.LotArea.median())[0])

# total_data.Street = total_data.Street.fillna('None')

# total_data.Alley = total_data.Alley.fillna("None")

# total_data.LotShape = total_data.LotShape.fillna('None')

# total_data.LandContour = total_data.LandContour.fillna('None')

# total_data.Utilities = total_data.Utilities.fillna('None')
cat_var = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition','YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']
for i in cat_var:

    total_data[i] = total_data[i].fillna('None')
total_data.MSSubClass = total_data.MSSubClass.fillna(0)

total_data.OverallQual = total_data.OverallQual.fillna(0)

total_data.OverallCond = total_data.OverallCond.fillna(0)
num_var = ['LotFrontage', 'LotArea','MasVnrArea','BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces',

       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold']
for i in num_var:

    print(i)

    total_data[i] = total_data[i].fillna(total_data[i].median())
total_data.head()
total_data.shape
cat_features = total_data.drop(['MSSubClass', 'LotFrontage', 'LotArea',

       'OverallQual', 'OverallCond','MasVnrArea','BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces',

       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold'],1)
cat_features.head()
num_features = total_data[['MSSubClass', 'LotFrontage', 'LotArea',

       'OverallQual', 'OverallCond','MasVnrArea','BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces',

       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'MiscVal',

       'MoSold']]
print(cat_features.shape)

print(num_features.shape)
cat_features.head()
num_features.head()
date_features = cat_features[['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']]

cat_features = cat_features.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],1)
date_features = pd.get_dummies(date_features, columns=['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'])

date_features.head()
cat_features.columns
cat_features = pd.get_dummies(cat_features, columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition'])

cat_features.head()
print(date_features.shape)

print(cat_features.shape)
from sklearn.preprocessing import MinMaxScaler

scalar_var = MinMaxScaler()
num_features = scalar_var.fit_transform(num_features)

print(num_features.shape)
num_features = pd.DataFrame(num_features)

num_features.head()
temp = pd.concat([cat_features,date_features],1)
print(temp.shape)

temp.head()
print(num_features.shape)

num_features.head()
np_temp = temp.values

np_num = num_features.values

print(np_temp.shape)

print(np_num.shape)
final_data = np.concatenate((np_temp,np_num),axis=1)
final_data = pd.DataFrame(final_data)
final_data.head()
final_data[final_data.columns[0]] = final_data[final_data.columns[0]].values.astype('int')
final_data.head()
final_data.shape
X_train = final_data.iloc[:1441]

X_test = final_data[1441:]

print(X_train.shape)

print(X_test.shape)
X_test.head()
X_train = X_train.drop(X_train.columns[0],1)

X_test = X_test.drop(X_test.columns[0],1)
X_train.shape
target = target.reshape(-1,1)
import xgboost as xgb
# reg_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

#                              learning_rate=0.05, max_depth=3, 

#                              min_child_weight=1.7817, n_estimators=2200,

#                              reg_alpha=0.4640, reg_lambda=0.8571,

#                              subsample=0.5213, silent=1,

#                              random_state =7, nthread = -1)
# reg.fit(X_train,target)
import lightgbm as lgb

reg_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
target = pd.DataFrame(target.reshape(-1,1))
target.head()
reg_lgb.fit(X_train,target)
# from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

# reg_gb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

#                                    max_depth=4, max_features='sqrt',

#                                    min_samples_leaf=15, min_samples_split=10, 

#                                    loss='huber', random_state =5)
# reg_ext = ExtraTreesRegressor(n_estimators=3000)
# from mlxtend.regressor import StackingRegressor
# stregr = StackingRegressor(regressors=[reg_xgb, reg_lgb, reg_gb], 

#                            meta_regressor=reg_ext)

# stregr.fit(X_train,target)

# reg_ext.fit(X_train, target)
# pred = stregr.predict(X_test)

pred = reg_lgb.predict(X_test)
pred = pred.reshape(-1,1)

print(pred.shape)
test_data.head()
_id = test_data.Id.values.reshape(-1,1)
_id.shape
type(_id[0])
output = np.array(np.concatenate((_id, pred), 1))
output = pd.DataFrame(output,columns = ["Id","SalePrice"])
output.head()
output.Id = output.Id.astype('Int64')
output.head()
output.to_csv('submission.csv',index = False)