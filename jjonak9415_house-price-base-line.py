import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)



%matplotlib inline

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# データがどのくらい欠損しているか確認



def missing_table(x):

    null_num = x.isnull().sum()

    percent = 100 * x.isnull().sum()/len(x)

    missing_table =  pd.concat([null_num,percent],axis=1)

    missing_table_ren_col = missing_table.rename(columns={0: '欠損値', 1: '%'})

    return missing_table_ren_col





missing_table(train)
missing_table(test)
train = train.drop(["Alley","PoolQC","Fence","MiscFeature"],axis=1)

missing_table(train)
test = test.drop(["Alley","PoolQC","Fence","MiscFeature"],axis=1)

missing_table(test)
v = train['GarageYrBlt'].mode()

v
from statistics import mode

train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())

train['MasVnrType'] = train['MasVnrType'].fillna(train['MasVnrType'].mode()[0])

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])

train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])

train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])

train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])

train['BsmtFinType1'] = train['BsmtFinType1'].fillna(train['BsmtFinType1'].mode()[0])

train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])

train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])

train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mode()[0])

train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])

train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])

train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
missing_table(train)
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].median())

test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])

test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])

test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])

test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])

test['BsmtExposure'] = test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])

test['BsmtFinType1'] = test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].median())

test['BsmtFinType2'] = test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])

test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mode()[0])

test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].median())

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median())

test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])

test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])



test['FireplaceQu'] = test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])

test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])

test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mode()[0])

test['GarageFinish'] = test['GarageFinish'].fillna(test['GarageFinish'].mode()[0])

test['GarageQual'] = test['GarageQual'].fillna(test['GarageQual'].mode()[0])

test['GarageCond'] = test['GarageCond'].fillna(test['GarageCond'].mode()[0])



test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
missing_table(test)
train
test
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()



def object_encode(arr):

    for x in arr:

        print('Now converting {} ....'.format(x))

        if arr[x].dtypes == 'object':

            arr[x][pd.isnull(arr[x])] = 'NaN' 

            arr[x] = label.fit_transform(arr[x])

            

object_encode(train)

object_encode(test)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

X = train.iloc[:,:-1].values

y = train['SalePrice'].values

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3)

model = LinearRegression()

model.fit(X_train,y_train)

y_train_pred = model.predict(X_train)

y_val_pred = model.predict(X_val)
from sklearn.metrics import  mean_squared_error

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

print('RMSE train: %.3f, validation: %.3f' %(rmse_train,rmse_val))
y_train
y_train_pred