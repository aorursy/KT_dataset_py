# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
data_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

print(data_train)



data_val = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_id = data_val['Id']
#For training data.



#Checking which features have what fraction of missing values

for i in range(0,data_train.shape[1]):

    for j in range(0, data_train.shape[0]):

        if(pd.isnull(data_train.iloc[j,i])):

            print(data_train.columns[i])

            print('Total missing values:',pd.isnull(data_train[data_train.columns[i]]).sum())

            print('Fraction of missing values',(pd.isnull(data_train[data_train.columns[i]]).sum())/data_train.shape[0])

            print('\n')

            break



#Imputing missing values appropriately.            

data_train['LotFrontage'].fillna(data_train['LotFrontage'].median(), inplace=True)

data_train['MasVnrType'].fillna(data_train['MasVnrType'].mode()[0], inplace=True)

data_train['MasVnrArea'].fillna(data_train['MasVnrArea'].median(), inplace=True)

data_train['BsmtQual'].fillna(data_train['BsmtQual'].mode()[0], inplace=True)

data_train['BsmtCond'].fillna(data_train['BsmtCond'].mode()[0], inplace=True)

data_train['BsmtExposure'].fillna(data_train['BsmtExposure'].mode()[0], inplace=True)

data_train['BsmtFinType1'].fillna(data_train['BsmtFinType1'].mode()[0], inplace=True)

data_train['BsmtFinType2'].fillna(data_train['BsmtFinType2'].mode()[0], inplace=True)

data_train['Electrical'].fillna(data_train['Electrical'].mode()[0], inplace=True)

data_train['GarageType'].fillna(data_train['GarageType'].mode()[0], inplace=True)

data_train['GarageFinish'].fillna(data_train['GarageFinish'].mode()[0], inplace=True)

data_train['GarageQual'].fillna(data_train['GarageQual'].mode()[0], inplace=True)

data_train['GarageCond'].fillna(data_train['GarageCond'].mode()[0], inplace=True)



print(data_train)    
#For validation data.

for i in range(0,data_val.shape[1]):

    for j in range(0, data_val.shape[0]):

        if(pd.isnull(data_val.iloc[j,i])):

            print(data_val.columns[i])

            print('Total missing values:',pd.isnull(data_val[data_val.columns[i]]).sum())

            print('Fraction of missing values',(pd.isnull(data_val[data_val.columns[i]]).sum())/data_val.shape[0])

            print('\n')

            break



data_val['MSZoning'].fillna(data_val['MSZoning'].mode()[0], inplace=True)

data_val['LotFrontage'].fillna(data_val['LotFrontage'].median(), inplace=True)

data_val['Utilities'].fillna(data_val['Utilities'].mode()[0], inplace=True)

data_val['Exterior1st'].fillna(data_val['Exterior1st'].mode()[0], inplace=True)

data_val['Exterior2nd'].fillna(data_val['Exterior2nd'].mode()[0], inplace=True)

data_val['MasVnrType'].fillna(data_val['MasVnrType'].mode()[0], inplace=True)

data_val['MasVnrArea'].fillna(data_val['MasVnrArea'].median(), inplace=True)

data_val['BsmtQual'].fillna(data_val['BsmtQual'].mode()[0], inplace=True)

data_val['BsmtCond'].fillna(data_val['BsmtCond'].mode()[0], inplace=True)

data_val['BsmtExposure'].fillna(data_val['BsmtExposure'].mode()[0], inplace=True)

data_val['BsmtFinType1'].fillna(data_val['BsmtFinType1'].mode()[0], inplace=True)

data_val['BsmtFinType2'].fillna(data_val['BsmtFinType2'].mode()[0], inplace=True)

data_val['BsmtFinSF1'].fillna(data_val['BsmtFinSF1'].median(), inplace=True)

data_val['BsmtFinSF2'].fillna(data_val['BsmtFinSF2'].median(), inplace=True)

data_val['BsmtUnfSF'].fillna(data_val['BsmtUnfSF'].median(), inplace=True)

data_val['TotalBsmtSF'].fillna(data_val['TotalBsmtSF'].median(), inplace=True)

data_val['BsmtFullBath'].fillna(data_val['BsmtFullBath'].median(), inplace=True)

data_val['BsmtHalfBath'].fillna(data_val['BsmtHalfBath'].median(), inplace=True)

data_val['KitchenQual'].fillna(data_val['KitchenQual'].mode()[0], inplace=True)

data_val['Functional'].fillna(data_val['Functional'].mode()[0], inplace=True)

data_val['GarageType'].fillna(data_val['GarageType'].mode()[0], inplace=True)

data_val['GarageFinish'].fillna(data_val['GarageFinish'].mode()[0], inplace=True)

data_val['GarageQual'].fillna(data_val['GarageQual'].mode()[0], inplace=True)

data_val['GarageCond'].fillna(data_val['GarageCond'].mode()[0], inplace=True)

data_val['GarageCars'].fillna(data_val['GarageCars'].median(), inplace=True)

data_val['GarageArea'].fillna(data_val['GarageArea'].median(), inplace=True)

data_val['SaleType'].fillna(data_val['SaleType'].mode()[0], inplace=True)



print(data_val)
#Creating new features using existing ones.

data_train['is_renovated']=1

data_train['is_renovated'].loc[data_train['YearBuilt']==data_train['YearRemodAdd']]=0

data_train['age_at_selling'] = data_train['YrSold'] - data_train['YearBuilt']

data_train['renovation_age'] = data_train['YrSold'] - data_train['YearRemodAdd']



data_val['is_renovated']=1

data_val['is_renovated'].loc[data_val['YearBuilt']==data_val['YearRemodAdd']]=0

data_val['age_at_selling'] = data_val['YrSold'] - data_val['YearBuilt']

data_val['renovation_age'] = data_val['YrSold'] - data_val['YearRemodAdd']



#Dropping features with high fraction of missing values.

drop_col = ['Id', 'GarageYrBlt', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

data_train.drop(drop_col, axis=1, inplace=True)

data_val.drop(drop_col, axis=1, inplace=True)



print(data_train.head())

print(data_val.head())

#Encoding categorical features.

label_encoder = LabelEncoder()

cont_col = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

           'LowQualFinSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

           'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice', 'age_at_selling', 'renovation_age']

for i in range(0, data_train.shape[1]):

    if(data_train.columns[i] not in cont_col):

        data_train[data_train.columns[i]] = label_encoder.fit_transform(data_train[data_train.columns[i]])



for i in range(0, data_val.shape[1]):

    if(data_val.columns[i] not in cont_col):

        data_val[data_val.columns[i]] = label_encoder.fit_transform(data_val[data_val.columns[i]])







print(data_train.head())

print(data_val.head())
y = data_train['SalePrice']

X = data_train.drop(['SalePrice'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



minimax = MinMaxScaler()

X_train = minimax.fit_transform(X_train)

X_test = minimax.transform(X_test)

data_val = minimax.transform(data_val)

#Linear Regression

linear_reg = LinearRegression()

linear_reg.fit(X_train, y_train)

linear_reg.score(X_test, y_test)
#Ridge Linear Regression(with L2 regularization)

ridge_reg = Ridge(alpha=10)

ridge_reg.fit(X_train, y_train)

ridge_reg.score(X_test, y_test)
#KNN Regression

knn_reg = KNeighborsRegressor(n_neighbors=10)

knn_reg.fit(X_train, y_train)

knn_reg.score(X_test, y_test)
#Elastic Net Regression(combined L1, L2 regularization)

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.7)

elastic_net.fit(X_train, y_train)

elastic_net.score(X_test, y_test)
predicted = pd.DataFrame()

predicted['Id'] = test_id

predicted['SalePrice'] = elastic_net.predict(data_val)

print(predicted)

predicted.to_csv('Submission.csv', index=False)