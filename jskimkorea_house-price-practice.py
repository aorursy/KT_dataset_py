# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample_submission=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
test.head()
num_col=train.select_dtypes(exclude='object')



num_col_corr=num_col.corr()



f, ax=plt.subplots(figsize=(30,30))

sns.heatmap(num_col_corr, annot=True, ax=ax)
num_col_corr['SalePrice']
high_corr_num_col=[]

for col in list(num_col_corr['SalePrice'].index):

    if (num_col_corr['SalePrice'][col]>0.5 or num_col_corr['SalePrice'][col]<-0.5):

        high_corr_num_col.append(col)
high_corr_num_col
object_col=train.select_dtypes('object')
object_col
f, axes=plt.subplots(9,5, figsize=(30,50))

ax=axes.ravel()

for i, col in enumerate(object_col.columns):

    sns.boxplot(x=object_col[col], y=train['SalePrice'], ax=ax[i])
high_corr_object_col=['SaleCondition','SaleType','MiscFeature','PoolQC','PavedDrive','KitchenQual','Electrical','CentralAir','BsmtQual','ExterQual','RoofMatl',

'Condition2','Neighborhood','Alley','MSZoning']
print(high_corr_num_col)

print(high_corr_object_col)
high_corr_num_col.remove('SalePrice')

corr_columns=high_corr_num_col+high_corr_object_col
train[corr_columns].head()
train[corr_columns].info()
y_train=train['SalePrice']
train=train[corr_columns]

test=test[corr_columns]



train.drop(columns=['MiscFeature','PoolQC','Alley'], inplace=True)

test.drop(columns=['MiscFeature','PoolQC','Alley'], inplace=True)
train.info()
test.info()
train_null_col=[col for col in train.columns if train[col].isnull().any()]

test_null_col=[col for col in test.columns if test[col].isnull().any()]
train_null_col
test_null_col
f, axes=plt.subplots(1,2,figsize=(20,4))

for i, col in enumerate(train_null_col):

    sns.countplot(train[col], ax=axes[i])
train['Electrical'].fillna('SBrkr', inplace=True)

train['BsmtQual'].fillna('TA', inplace=True)
f, axes=plt.subplots(3,3, figsize=(20,20))

ax=axes.ravel()

for i, col in enumerate(test_null_col):

    if test[col].dtype=='int64' or test[col].dtype=='float64':

        sns.distplot(test[col].dropna(), ax=ax[i])

    else:

        sns.countplot(test[col].dropna(), ax=ax[i])
test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].median(), inplace=True)

test['GarageCars'].fillna(2, inplace=True)

test['GarageArea'].fillna(test['GarageArea'].median(), inplace=True)

for col in ['SaleType','KitchenQual','BsmtQual','MSZoning']:

    test[col].fillna(test[col].value_counts().index[0], inplace=True)
test.info()
train.info()
train.columns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import KFold, GridSearchCV



ct_passthrough=ColumnTransformer([('onehotencoding', OneHotEncoder(handle_unknown='ignore'), ['SaleCondition', 'SaleType', 'PavedDrive', 'KitchenQual', 'Electrical',

       'CentralAir', 'BsmtQual', 'ExterQual', 'RoofMatl', 'Condition2','Neighborhood', 'MSZoning']),

                     ('pass', 'passthrough', ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',

       'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']),

                     ], n_jobs=-1)



ct_MinMaxScaling=ColumnTransformer([('onehotencoding', OneHotEncoder(handle_unknown='ignore'), ['SaleCondition', 'SaleType', 'PavedDrive', 'KitchenQual', 'Electrical',

       'CentralAir', 'BsmtQual', 'ExterQual', 'RoofMatl', 'Condition2','Neighborhood', 'MSZoning']),

                     ('pass', MinMaxScaler(), ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',

       'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']),

                     ], n_jobs=-1)
from sklearn.pipeline import Pipeline, make_pipeline



pipe=Pipeline([('preprocessing', ct_passthrough),

              ('Regressor', LinearRegression())])



param_grid=[{'Regressor':[LinearRegression()], 'preprocessing':[ct_passthrough]},

           {'Regressor':[SVR()], 'preprocessing':[ct_MinMaxScaling]},

           {'Regressor':[RandomForestRegressor()], 'preprocessing':[ct_passthrough]}]



GS=GridSearchCV(pipe, param_grid=param_grid, cv=KFold(), n_jobs=-1)
GS.fit(train, np.log(y_train))
predict=GS.predict(test)
predict
np.exp(predict)
sub=sample_submission

sub['SalePrice']=np.exp(predict)

sub.to_csv('submission.csv', index=False)