# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")  

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex1 import *

print("Setup Complete")
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Read the data

train_data = pd.read_csv('../input/train.csv', index_col='Id')

test_data = pd.read_csv('../input/test.csv', index_col='Id')
corrmat = train_data.corr(method='spearman')

f, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corrmat, ax=ax, annot=True, fmt=".1f", annot_kws={'size':8}, center=0, linewidths=0.1)
features = ['LotArea', 'LotFrontage', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF','MasVnrArea', 'OverallQual', 'FullBath', 'HalfBath',

            'BedroomAbvGr', 'TotRmsAbvGrd', 'GrLivArea', 'GarageArea', 'Fireplaces', 'OpenPorchSF', 'WoodDeckSF', 'TotalBsmtSF', 'BsmtFinSF1']
condition = ['KitchenQual', 'PoolQC', 'GarageCond', 'BsmtCond', 'FireplaceQu', 'HeatingQC', 'BsmtQual', 'GarageQual', 'ExterQual', 'ExterCond']



for i in condition:

    train_nan = train_data.loc[:,i].isna()

    test_nan = test_data.loc[:,i].isna()

    for j in train_data.index:

        if train_data.loc[j,i]=='Gd':

            train_data.loc[j,i]=4

        elif train_data.loc[j, i]=='Po':

            train_data.loc[j,i]=1

        elif train_data.loc[j, i]=='Ex':

            train_data.loc[j, i]=5

        elif train_data.loc[j, i]=='Fa':

            train_data.loc[j, i]=2

        elif train_data.loc[j, i]=='TA':

            train_data.loc[j, i]=3

        elif train_data.loc[j, i]=='NA' or train_data.loc[j, i]=='na' or train_nan[j]==True:

            train_data.loc[j, i]=0

    for z in test_data.index:

        if test_data.loc[z,i]=='Gd':

            test_data.loc[z,i]=4

        elif test_data.loc[z, i]=='Po':

            test_data.loc[z,i]=1

        elif test_data.loc[z, i]=='Ex':

            test_data.loc[z, i]=5

        elif test_data.loc[z, i]=='Fa':

            test_data.loc[z, i]=2

        elif test_data.loc[z, i]=='TA':

            test_data.loc[z, i]=3

        elif test_data.loc[z, i]=='NA' or test_data.loc[z, i]=='na' or test_nan[z]==True:

            test_data.loc[z, i]=0
features.extend(condition)



corr_check = features

corr_check.append('SalePrice')



for i in condition:

    train_data[i] = train_data[i].astype(float)

    test_data[i] = test_data[i].astype(float)



train_reduced = train_data[corr_check]

corrmat = train_reduced.corr(method='spearman')

f, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(corrmat, ax=ax, annot=True, fmt=".1f", annot_kws={'size':8}, center=0, linewidths=0.1)



features.remove('SalePrice')
features.remove('PoolQC')

features.remove('ExterCond')



y = train_data['SalePrice']

X = train_data[features]

X_realtest = test_data[features]



print(X.isnull().sum())

print(X_realtest.isnull().sum())
values = {'LotFrontage': 0, 'MasVnrArea': 0}

test_values = {'GarageArea':0, 'TotalBsmtSF': X_realtest['TotalBsmtSF'].mean(), 'BsmtFinSF1': X_realtest['BsmtFinSF1'].mean()}

X.fillna(value=values, inplace=True)

X_realtest.fillna(value=values, inplace=True)

X_realtest.fillna(value=test_values, inplace=True)
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model1 = RandomForestRegressor()

model2 = GaussianProcessRegressor()

model3 = AdaBoostRegressor()

model4 = GradientBoostingRegressor()

model5 = XGBRegressor()
models = [model1, model2, model3, model4, model5]



def score_model(model, X_t=X_train, X_v=X_test, y_t=y_train, y_v=y_test):

    model.fit(X_t, y_t)

    y_pred = model.predict(X_v)

    return mean_squared_log_error(y_v, y_pred)



for model in models:

    print(score_model(model))