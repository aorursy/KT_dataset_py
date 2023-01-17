import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

test_sub = pd.read_csv("../input/test.csv")

houses_data = pd.concat([train,test], sort = False)

print(train.shape)

print(test.shape)
houses_data.describe()
missing_val_count_by_column = (houses_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
houses_data.select_dtypes(include=['object']).isnull().sum()[houses_data.select_dtypes(include=['object']).isnull().sum()>0]
for col in  ('Alley','Utilities','Exterior1st','Exterior2nd','MasVnrType','BsmtQual','BsmtCond',

             'BsmtExposure','BsmtFinType1','BsmtFinType2','KitchenQual','FireplaceQu','GarageType',

             'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'):

    train[col]=train[col].fillna('None')

    test[col]=test[col].fillna('None')
for col in  ('MSZoning','Electrical','Functional','SaleType'):

    train[col]=train[col].fillna(train[col].mode()[0])

    test[col]=test[col].fillna(train[col].mode()[0])
#Model_Prep

house_data = pd.concat([train, test], sort = False)

house_data = pd.get_dummies(house_data)

len_train = train.shape[0]

train = house_data[:len_train]

test = house_data[len_train:]

print(train.shape, test.shape, house_data.shape)
#drop_Id

train.drop('Id', axis=1, inplace=True)

test.drop('Id', axis=1, inplace=True)
X = train.drop('SalePrice', axis=1) 

y = train['SalePrice'] 

test = test.drop('SalePrice', axis=1)
from xgboost import XGBRegressor

import xgboost as xgb

#Matrix

house_matrix = xgb.DMatrix(data = X, label = y)

house_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

house_model.fit(X, y, verbose=False)
# make predictions

predictions = house_model.predict(test)
output = pd.DataFrame({'Id': test_sub.Id,'SalePrice': predictions}) 

output.to_csv('submission.csv', index=False)

output.head()