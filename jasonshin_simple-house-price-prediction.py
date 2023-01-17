import numpy as np

import pandas as pd

import scipy.stats   

import scipy.special  

import subprocess

import sklearn.linear_model

import sklearn.model_selection

import sklearn.pipeline  

import sklearn.preprocessing

import sklearn.ensemble  

import sklearn.kernel_ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Ridge, Lasso

from sklearn.linear_model import LinearRegression

from mlxtend.regressor import StackingRegressor



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



color = sns.color_palette()

sns.set_style('darkgrid')



pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 100)
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/train.csv")

df = pd.concat([train_df, test_df])

print('train_df shape: ', train_df.shape)

print('test_df shape: ', test_df.shape)

print('df shape: ', df.shape)
df.columns
df.dtypes
print(train_df['SalePrice'].describe())



print('Medium value is ', train_df['SalePrice'].median())

sns.distplot(train_df['SalePrice'], fit=scipy.stats.norm)
print(df.isnull().sum())

cols_with_missing = [col for col in df.columns 

                                 if df[col].isnull().any()]

print("Columns with missing features: ",  cols_with_missing)
imputer = sklearn.preprocessing.Imputer()
# Drop columns where large number of values are null

columns_to_drop = ['PoolQC', 'Fence', 'MiscFeature', 'Alley']

df = df.drop(columns_to_drop, axis=1)
from sklearn.preprocessing import Imputer

my_imputer = Imputer()

df[['LotFrontage']] = my_imputer.fit_transform(df[['LotFrontage']])
print(df.isnull().sum())
df[['MasVnrType']] = df[['MasVnrType']].fillna(value='None')

df[['MasVnrArea']] = df[['MasVnrArea']].fillna(value=0)



df[['BsmtQual']] = df[['BsmtQual']].fillna(value='None')

df[['BsmtCond']] = df[['BsmtCond']].fillna(value=0)



df[['BsmtExposure']] = df[['BsmtExposure']].fillna(value='NA')

df[['BsmtFinType1']] = df[['BsmtFinType1']].fillna(value='NA')

df[['BsmtFinType2']] = df[['BsmtFinType2']].fillna(value='NA')



# Only 2 missing values, drop it

df = df.dropna(subset=['Electrical'], how='all')



df[['FireplaceQu']] = df[['FireplaceQu']].fillna(value='NA')



df[['GarageType']] = df[['GarageType']].fillna(value='NA')

df[['GarageFinish']] = df[['GarageFinish']].fillna(value='NA')

df[['GarageQual']] = df[['GarageQual']].fillna(value='NA')

df[['GarageCond']] = df[['GarageCond']].fillna(value='NA')

df[['GarageYrBlt']] = df[['GarageYrBlt']].fillna(value=0)
df['MSSubClass'] = df['MSSubClass'].apply(str)

df['OverallCond'] = df['OverallCond'].astype(str)

df['YrSold'] = df['YrSold'].astype(str)

df['MoSold'] = df['MoSold'].astype(str)
# categorial features

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



# apply sklearn.preprocessing.LabelEncoder to each categorical feature

for c in cols:

    lbl = sklearn.preprocessing.LabelEncoder() 

    lbl.fit(list(df[c].values)) 

    df[c] = lbl.transform(list(df[c].values))



# shape        

print('data_df.shape = ', df.shape)
cols = df.select_dtypes(exclude =[np.number]).columns.values

df  = pd.get_dummies(df).copy()
train_df = df[: train_df.shape[0]]

train_df_y = train_df[['SalePrice']]

train_df = train_df.drop('SalePrice', axis=1)
test_df = df[train_df.shape[0]-1:]

test_df = test_df.drop('SalePrice', axis=1)
test_df.shape
test_df.loc[-1:]
# Initialize models

lr = LinearRegression(

    n_jobs = -1

)



rd = Ridge(

    alpha = 4.84

)



rf = RandomForestRegressor(

    n_estimators = 12,

    max_depth = 3,

    n_jobs = -1

)



gb = GradientBoostingRegressor(

    n_estimators = 40,

    max_depth = 2

)



nn = MLPRegressor(

    hidden_layer_sizes = (90, 90),

    alpha = 2.75

)

# Initialize Ensemble

model = StackingRegressor(

    regressors=[rf, gb, nn, rd],

    meta_regressor=Lasso(copy_X=True, fit_intercept=True, normalize=False)

)
model.fit(train_df, train_df_y.values.ravel())
predictions = model.predict(test_df)
submission = pd.DataFrame()

# Sets up proper indexes

indexs = [i for i in range(1461, 2920)]

submission['Id'] = indexs

submission['SalePrice'] = predictions

submission.to_csv('submission.csv', index=False)