import numpy as np

import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



train.head()
#Train Data - Check for Null Values 

for column in train.columns:

    if(train[column].isnull().sum() != 0):

        print('Feature : ', column , ' ------ No of nulls : ',train[column].isnull().sum(), ' ----- Per : ',train[column].isnull().sum()*100/1460)
train = train.drop(columns=['LotFrontage','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1)



test = test.drop(columns=['LotFrontage','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1)
for column in train.columns:

    if(train[column].isnull().sum() != 0):

        print('Feature : ', column , ' # No of nulls : ', train[column].isnull().sum(), ' # Per : ', train[column].isnull().sum()*100/1460)
train = train.drop(columns=['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 

                            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MasVnrArea'], 

                            axis=1)



test = test.drop(columns=['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 

                            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MasVnrArea'], 

                            axis=1)
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])



#Test dataset doesnot contain any Null value for 'Eletrical' column

#print(test['Electrical'].isnull().sum()) 

#give 0 as result
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(20, 20))



sns.set(font_scale=1)

sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
#SalePrice Correlation with following columns is less than 10%. Let's drop these columns.

train = train.drop(columns=['MSSubClass', 'OverallCond', 'BsmtHalfBath', 'BsmtUnfSF', 'BsmtFinSF2',

                              'LowQualFinSF', '3SsnPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'], axis=1)



test = test.drop(columns=['MSSubClass', 'OverallCond', 'BsmtHalfBath', 'BsmtUnfSF', 'BsmtFinSF2',

                              'LowQualFinSF', '3SsnPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'], axis=1)
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(20, 20))



sns.set(font_scale=1)

sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
#bivariate analysis saleprice/overallqual

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

train.loc[(train['OverallQual'] == 10) & (train['SalePrice'] < 200000)]



train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)
#bivariate analysis saleprice/totalbsmtsf

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#bivariate analysis saleprice/1stflrsf

var = '1stFlrSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

train.sort_values(by = '1stFlrSF', ascending = False)[:2]



train = train.drop(train[train['Id'] == 497].index)

train = train.drop(train[train['Id'] == 1025].index)
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#bivariate analysis saleprice/overallqual

var = 'GarageCars'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#bivariate analysis saleprice/garagearea

var = 'GarageArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

train.sort_values(by = 'GarageArea', ascending = False)[:3]



train = train.drop(train[train['Id'] == 582].index)

train = train.drop(train[train['Id'] == 1062].index)

train = train.drop(train[train['Id'] == 1191].index)
#Checking Null values in Test Dataset

test.isnull().sum()
#Filling null values with Mode

for column in test.columns:

    if(test[column].isnull().sum() != 0):

        test[column].fillna(test[column].mode()[0], inplace=True)
train['train']  = 1

test['train']  = 0

df = pd.concat([train, test], axis=0, sort=False)

print(df.shape)
#Convert categorical variable into dummy

df = pd.get_dummies(df)

df.head()
df_final = df.drop(['Id'], axis=1)



df_train = df_final[df_final['train'] == 1]

df_train = df_train.drop(['train'], axis=1)



df_test = df_final[df_final['train'] == 0]

df_test = df_test.drop(['train'], axis=1)

df_test = df_test.drop(['SalePrice'], axis=1)
target = df_train['SalePrice']

df_train = df_train.drop(['SalePrice'],axis=1)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train, target, test_size=0.2, random_state=0)
from lightgbm import LGBMRegressor

import sklearn.metrics as metrics

import math



lgbm2 = LGBMRegressor(objective='regression', 

                                       num_leaves=8,

                                       learning_rate=0.005, 

                                       n_estimators=15000, 

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.4, 

                                       )



lgbm2.fit(x_train, y_train, eval_metric='rmse')



lgbm2_pred = lgbm2.predict(x_test)



print('Root Mean Square Error test = ' + str(math.sqrt(metrics.mean_squared_error(y_test, lgbm2_pred))))
lgbm2.fit(df_train, target, eval_metric='rmse')

lgbm2_pred_allTest = lgbm2.predict(df_test)



submission = pd.DataFrame({"Id": test["Id"], "SalePrice": lgbm2_pred_allTest * 2})

submission.to_csv('Final_submission_best.csv', index=False)