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
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col = 'Id')
train_data
train_data.info()
cols_with_missing = [col for col in train_data.columns
                     if train_data[col].isnull().any()]
print(cols_with_missing)
for col in cols_with_missing:
    print (col, train_data[col].isnull().sum() / len(train_data))
cols_with_big_amount_missing = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
reduced_train_data = train_data.drop(cols_with_big_amount_missing, axis=1)
cols_with_missing = [col for col in reduced_train_data.columns
                     if train_data[col].isnull().any()]
cols_with_missing
reduced_train_data.info()
for col in cols_with_missing:
    print (train_data[col].isnull().sum())
object_columns_with_missing_values = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                                     'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish',
                                     'GarageQual', 'GarageCond']
numeric_columns_with_missing_values = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
reduced_train_data.dropna(axis=0, inplace=True)
reduced_train_data
for col in object_columns_with_missing_values:
    print (reduced_train_data[col].value_counts().idxmax())
for col in object_columns_with_missing_values: 
    reduced_train_data[col].fillna(value = reduced_train_data[col].value_counts().idxmax(), inplace = True)
for col in numeric_columns_with_missing_values:
    reduced_train_data[col].fillna(value = reduced_train_data[col].mean(), inplace = True)
reduced_train_data['MasVnrType'].unique()
reduced_train_data.info()

s = (reduced_train_data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
from sklearn.preprocessing import LabelEncoder
label_reduced_train_data = reduced_train_data.copy()
# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_reduced_train_data[col] = label_encoder.fit_transform(reduced_train_data[col])
label_reduced_train_data.info()
target_col = 'SalePrice'
y = label_reduced_train_data[target_col]
X = label_reduced_train_data.drop(columns=[target_col])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
para = list(range(50, 1001, 100))
results = {}
for n in para:
    print('para=', n)
    model = RandomForestRegressor(n_estimators=n, random_state = 0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print (mae)
    results[n] = mae


test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
test_data.head()
train_data.info()
test_data.info()
cols_with_missing_test = [col for col in test_data.columns
                     if test_data[col].isnull().any()]
cols_with_missing_test
for col in cols_with_missing_test:
    print (test_data[col].isnull().sum())
cols_with_big_amount_missing = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
reduced_test_data = test_data.drop(cols_with_big_amount_missing, axis=1)
reduced_test_data.info()
cols_with_missing_test = [col for col in reduced_test_data.columns
                     if reduced_test_data[col].isnull().any()]
cols_with_missing_test
for col in cols_with_missing_test:
    print (reduced_test_data[col].isnull().sum())
object_columns_test_with_missing_values = ['MasVnrType', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 
                                           'BsmtFinType2', 'GarageType', 'GarageFinish','GarageQual', 'GarageCond',
                                           'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
                                           'KitchenQual', 'Functional', 'SaleType', 'FireplaceQu']
numeric_columns_test_with_missing_values = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                                            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 
                                            'GarageCars', 'GarageArea',]
for col in object_columns_test_with_missing_values:
    print (reduced_train_data[col].value_counts().idxmax())
for col in object_columns_test_with_missing_values: 
    reduced_test_data[col].fillna(value = reduced_train_data[col].value_counts().idxmax(), inplace = True)
for col in numeric_columns_test_with_missing_values:
    reduced_test_data[col].fillna(value = reduced_train_data[col].mean(), inplace = True)
reduced_test_data.info()
s = (reduced_test_data.dtypes == 'object')
object_cols_test = list(s[s].index)

print("Categorical variables:")
print(object_cols_test)
print (object_cols)
from sklearn.preprocessing import LabelEncoder
label_reduced_test_data = reduced_test_data.copy()
# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols_test:
    label_reduced_train_data[col] = label_encoder.fit_transform(reduced_train_data[col].astype(str))
    label_reduced_test_data[col] = label_encoder.transform(reduced_test_data[col].astype(str))
    