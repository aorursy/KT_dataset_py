# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# To develop plots

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# For normality transformation

from scipy import stats

from scipy.stats import norm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.head()
df_train.info()
df_train.describe()
df_train.isnull().sum().sort_values(ascending=False).head(20)
df_train = df_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)

df_test = df_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, square=True);
df_train = df_train.drop(['MasVnrArea', 'MasVnrType'], axis=1)

df_test = df_test.drop(['MasVnrArea', 'MasVnrType'], axis=1)



df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_test = df_test.drop(df_test.loc[df_test['Electrical'].isnull()].index)



cols = ['GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure',

        'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']

for col in cols:

    df_train[col] = df_train[col].fillna('None')

    df_test[col] = df_test[col].fillna('None')

# To check if there is still any missing data

df_train.isnull().sum().max()
X = df_train.drop(['SalePrice'], axis=1)

y = df_train['SalePrice']
sns.distplot(y, fit=norm);

fig = plt.figure()

res = stats.probplot(y, plot=plt)
y = np.log(y)
sns.distplot(y, fit=norm);

fig = plt.figure()

res = stats.probplot(y, plot=plt)
len(X.select_dtypes(include=['object']).columns)
X['TotalBath'] = (X['FullBath'] + (0.5 * X['HalfBath']) +

                  X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))

df_test['TotalBath'] = (df_test['FullBath'] + (0.5 * df_test['HalfBath']) +

                  df_test['BsmtFullBath'] + (0.5 * df_test['BsmtHalfBath']))



X = X.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)

df_test = df_test.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1)
X.drop(['Id'],axis=1, inplace=True)

df_test_res = pd.DataFrame(columns=['Id', 'SalePrice'])

df_test_res['Id'] = df_test['Id']

df_test.drop(['Id'],axis=1, inplace=True)



X.head()
X = pd.get_dummies(X.drop(['GarageYrBlt'],axis=1))

df_test = pd.get_dummies(df_test.drop(['GarageYrBlt'],axis=1))



X.head()
# Handling missing data in Testing dataset

for col in df_test.columns:

    if col not in X_train.columns:

        df_test.drop([col],axis=1, inplace=True)

for col in X_train.columns:

    if col not in df_test.columns:

        X_train.drop([col],axis=1, inplace=True)



print(X_train.shape)

print(df_test.shape)
from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

import math



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)



parameters = [{'alpha': [0.2, 0.4, 0.6, 0.8, 1.0], 'max_iter': [1e7]}]



clf = GridSearchCV(Lasso(), parameters, cv=5,

                       scoring='neg_mean_squared_error')



clf.fit(X_train, y_train)



clf.best_params_



y_true, y_pred = y_test, clf.predict(X_test)



math.sqrt(mean_squared_error(y_true, y_pred))
# df_test_res['SalePrice'] = np.exp(clf.predict(df_test))