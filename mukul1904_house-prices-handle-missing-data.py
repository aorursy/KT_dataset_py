import numpy as np

import pandas as pd

import os

import sys

import tqdm

from multiprocessing import  Pool

import warnings

warnings.filterwarnings("ignore")

from math import sqrt

train_on_gpu = False



# Visualisation libs

import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerLine2D

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

from sklearn.decomposition import PCA

from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor
print('In input directory:')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sample_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')



train.shape, test.shape, sample_submission.shape
def score(y_actual, y_predicted):

    # because competetion uses RMSLE

    return sqrt(mean_squared_log_error(y_actual, y_predicted))

    

def fillNaNInfinity(df):

    df.replace([np.inf, -np.inf], np.nan)

    df.fillna(0, inplace=True)

    return df



def fillInfinity(df):

    df.replace([np.inf, -np.inf], np.nan)

    return df
data = pd.concat([

    train.loc[:, train.columns != 'SalePrice'], test

])



target = np.log(train['SalePrice'] + 1)



data.shape, target.shape
# From https://www.kaggle.com/miguelangelnieto/pca-and-regression#Simple-Neural-Network, loved it

nans = pd.isnull(data).sum()

nans[ nans > 0 ]
columns_to_remove = nans[ nans > 500 ].reset_index()['index'].tolist()

columns_to_remove
data.drop(labels=columns_to_remove, axis=1, inplace=True)

data.shape
nans = pd.isnull(data).sum()

nans[ nans > 5 ]
df = data.copy() # don't want to modify orginal data



more_columns_to_remove = nans[ nans > 5 ].reset_index()['index'].tolist()

df.drop(labels=more_columns_to_remove, axis=1, inplace=True)

print(df.shape)

df = pd.get_dummies(df)

print(df.shape)

xgbr = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7)

X = df[:1460]

y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)



xgbr.fit(X_train, y_train)



score(np.exp(y_train) - 1, np.exp(xgbr.predict(X_train)) - 1), score(np.exp(y_test) - 1, np.exp(xgbr.predict(X_test)) - 1)
data['LotFrontage'].describe()
sns.distplot(data['LotFrontage'].fillna(0));
data['Neighborhood'].describe()
data.groupby('Neighborhood')['LotFrontage'].agg(['count', 'mean', 'median'])
data[ data['LotFrontage'].isnull() ]['Neighborhood'].reset_index()['Neighborhood'].isnull().sum()
# https://stackoverflow.com/questions/39480997/how-to-use-a-user-function-to-fillna-in-pandas

data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda group: group.fillna(group.median()))



data['LotFrontage'].isnull().sum()
data['MasVnrType'].describe()
data['MasVnrType'].value_counts()
data['MasVnrArea'].describe()
sns.distplot(data['MasVnrArea'].fillna(0));
data.groupby('MasVnrType')['MasVnrArea'].agg(['mean', 'median', 'count'])
# data.drop(labels=['MasVnrArea'], axis=1, inplace=True)



# I checked that model works . better if we keep these 2 columns
data[['BsmtCond', 'BsmtQual']].describe()
data[['BsmtCond', 'BsmtQual']].isnull().sum()
data['BsmtCond'].value_counts()
data['BsmtQual'].value_counts()
# NA is not available

data[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].fillna('NA', inplace=True)
data[[

    'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual' 

]].describe()
df = data.copy()

df = pd.get_dummies(df)

df.shape
xgbr = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7)

X = df[:1460]

y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)



xgbr.fit(X_train, y_train)
score(np.exp(y_train) - 1, np.exp(xgbr.predict(X_train)) - 1), score(np.exp(y_test) - 1, np.exp(xgbr.predict(X_test)) - 1)
test = df[1460:]

sample_submission['SalePrice'] = xgbr.predict(test)

sample_submission['SalePrice'] = np.exp(sample_submission['SalePrice']) - 1

sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)