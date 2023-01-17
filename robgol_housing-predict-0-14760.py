# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb

# sklearn imports

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, Normalizer, OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier, XGBRegressor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedShuffleSplit,train_test_split, ShuffleSplit

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error



# Keras imports 

from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.losses import mean_squared_logarithmic_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# loading and verifying the shapes

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

df_train.shape, df_test.shape, df_sample_submission.shape
# the pathern to submit in kaggle

df_sample_submission.head()
#spliting the target and join the data

target = df_train['SalePrice']

df_train = df_train.drop('SalePrice', axis = 1)

df_train_test = pd.concat([df_train, df_test])

print(f'train_test_shape: {df_train_test.shape}')
# checking NaN values in traintest dataframe

import operator

NaNColumns_train_test = {}

for cols in df_train_test.columns:

    if df_train_test[cols].isna().sum() > 0:

        NaNColumns_train_test[cols] = df_train_test[cols].isna().sum()

sorted_NaNColumns_train_test = sorted(NaNColumns_train_test.items(), key=operator.itemgetter(1), reverse = True)

print(sorted_NaNColumns_train_test)

print([i[0] for i in sorted_NaNColumns_train_test])

        
train_test_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'MasVnrArea', 'MSZoning', 'Utilities', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'KitchenQual', 'GarageCars', 'GarageArea', 'SaleType']



for i in train_test_columns:

    df_train_test[i] = pd.Categorical(df_train_test[i]).codes
object_columns = [df_train_test.dtypes.keys()[i] for i, j in enumerate(df_train_test.dtypes.values) if j =='O']

for i in object_columns:

    print(f'For {i} columns there is {len(df_train_test[i].value_counts())} diferent values!')


for i in object_columns:

    print(f'The index of {i} is - {list(df_train_test.columns).index(i)}')

    index_stored.append(list(df_train_test.columns).index(i))

# Let's generate dummies variables here

df_train_test = pd.get_dummies(df_train_test)

df_train_test.shape
X = df_train_test.iloc[:df_train.shape[0], :].values

X_teste = df_train_test.iloc[df_train.shape[0]:, :].values

y
# my first submission with data transform 

random = RandomForestRegressor(n_estimators=1000)

y = y.ravel()

random.fit(X, y)

p = random.predict(X_teste)

## Two forms to generate data to submit

random_submission = pd.DataFrame({'Id': df_test['Id'].values, 'SalePrice': p})

serie_submission = pd.Series(p, index = df_test['Id'], name = 'SalePrice')



serie_submission.to_csv('submission_serie01.csv', header = True)

random_submission.to_csv('submission01.csv', index = False)