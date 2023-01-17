%matplotlib inline

import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.api.types import is_numeric_dtype



sns.set()
rawtrain = pd.read_csv('../input/train.csv')

rawtest = pd.read_csv('../input/test.csv')
print('Train shape:', rawtrain.shape)

print('Test shape:', rawtest.shape)
rawtrain.dtypes.value_counts()
selected = ['GrLivArea',

 'LotArea',

 'BsmtUnfSF',

 '1stFlrSF',

 'TotalBsmtSF',

 'GarageArea',

 'BsmtFinSF1',

 'LotFrontage',

 'YearBuilt',

 'Neighborhood',

 'GarageYrBlt',

 'OpenPorchSF',

 'YearRemodAdd',

 'WoodDeckSF',

 'MoSold',

 '2ndFlrSF',

 'OverallCond',

 'Exterior1st',

 'YrSold',

 'OverallQual']
#features = [c for c in test.columns if c not in ['Id']]
train = rawtrain[selected].copy()

train['is_train'] = 1

train['SalePrice'] = rawtrain['SalePrice'].values

train['Id'] = rawtrain['Id'].values



test = rawtest[selected].copy()

test['is_train'] = 0

test['SalePrice'] = 1  #dummy value

test['Id'] = rawtest['Id'].values



full = pd.concat([train, test])



not_features = ['Id', 'SalePrice', 'is_train']

features = [c for c in train.columns if c not in not_features]
def summary(df, dtype):

    data = []

    for c in df.select_dtypes([dtype]).columns:

        data.append({'name': c, 'unique': df[c].nunique(), 

                     'nulls': df[c].isnull().sum(),

                     'samples': df[c].unique()[:20] })

    return pd.DataFrame(data)
#for c in full.columns:

#    assert full[c].isnull().sum() == 0, f'There are still missing values in {c}'
#for c in full.columns:

#    assert is_numeric_dtype(full[c]), f'Non-numeric column {c}'
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

from sklearn.model_selection import train_test_split

from sklearn import metrics
def rmse(y_true, y_pred):

    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))
#subm = pd.DataFrame(ypred, index=test['Id'], columns=['SalePrice'])

#subm.to_csv('submission.csv')
#subm = pd.DataFrame(blendtestpred, index=test['Id'], columns=['SalePrice'])

#subm.to_csv('submission_blend.csv')