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

cols_with_missing
for col in cols_with_missing:

    print (train_data[col].isnull().sum())
cols_with_big_amount_missing = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']

reduced_train_data = train_data.drop(cols_with_big_amount_missing, axis=1)
s = (reduced_train_data.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)
object_columns_with_missing_values = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

                                     'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish',

                                     'GarageQual', 'GarageCond']

numeric_columns_with_missing_values = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
for col in object_columns_with_missing_values:

    print (col, reduced_train_data[col].value_counts().idxmax())
for col in object_columns_with_missing_values: 

    reduced_train_data[col].fillna(value = reduced_train_data[col].value_counts().idxmax(), inplace = True)

for col in numeric_columns_with_missing_values:

    reduced_train_data[col].fillna(value = reduced_train_data[col].median(), inplace = True)
from sklearn.preprocessing import LabelEncoder

label_reduced_train_data = reduced_train_data.copy()

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_reduced_train_data[col] = label_encoder.fit_transform(reduced_train_data[col])
cols_with_missing = [col for col in reduced_train_data.columns

                     if train_data[col].isnull().any()]

cols_with_missing
reduced_train_data.info()
target_col = 'SalePrice'

y = label_reduced_train_data[target_col]

X = label_reduced_train_data.drop(columns=[target_col])

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)
for col in cols_with_missing:

    print (train_data[col].isnull().sum())
n_estimators = list(range(100, 1001, 100)), 

learning_rate = [x / 100 for x in range(5, 100, 10)]
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

preds_dict = []

for n_estimators in range (100,1001,100):

    for max_depth in range (6, 70,10):

        parameters = {'n_estimators': n_estimators,

                          'max_depth': max_depth, 

                          'learning_rate': 0.05

                         }

        xgb = XGBRegressor(**parameters)

        xgb.fit(X_train,y_train)

        predictions = xgb.predict(X_valid)

        prediction = {}

        prediction['max_depth'] = max_depth

        prediction['n_estimators'] = n_estimators

        prediction['MAE'] = mean_absolute_error(y_valid,predictions)

        preds_dict.append(prediction)

print (preds_dict)
train_data
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
for col in object_columns_test_with_missing_values: 

    reduced_test_data[col].fillna(value = train_data[col].value_counts().idxmax(), inplace = True)

for col in numeric_columns_test_with_missing_values:

    reduced_test_data[col].fillna(value = train_data[col].mean(), inplace = True)
from sklearn.preprocessing import LabelEncoder

label_reduced_test_data = reduced_test_data.copy()

# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_reduced_train_data[col] = label_encoder.fit(train_data[col].astype(str))

    label_reduced_test_data[col] = label_encoder.transform(reduced_test_data[col].astype(str))

    

    
s = (reduced_test_data.dtypes == 'object')

object_cols = list(s[s].index)

print("Categorical variables:")

print(object_cols)
final_model = XGBRegressor(n_estimators=900, 

                          learning_rate=0.05, 

                          max_depth=1000)
final_model.fit(X, y)
preds_test = final_model.predict(label_reduced_test_data)
output = pd.DataFrame({'Id': label_reduced_test_data.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)

print("done")