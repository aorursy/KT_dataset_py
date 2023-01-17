# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
data.head()
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice

X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)
from xgboost import XGBRegressor



xgboost = XGBRegressor()

xgboost.fit(train_X, train_y, verbose=False)
predictions = xgboost.predict(test_X)
from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
xgbregressor = XGBRegressor(n_estimators=1000)

xgbregressor.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)
xgbregressor = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgbregressor.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)
data.describe()
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_data.columns
train_data_y = train_data.SalePrice
train_data1 = train_data.drop(['Id', 'SalePrice'], axis = 1)

test_data1 = test_data.drop(['Id'], axis = 1)
common_columns = train_data1.columns.intersection(test_data.columns)

train_data1 = train_data1[common_columns]

test_data1 = test_data1[common_columns]

from sklearn.impute import SimpleImputer



train_data_numeric = train_data1.select_dtypes(exclude=['object'])

train_data_numeric_columns = train_data_numeric.columns

my_imputer = SimpleImputer()

train_data_numeric = my_imputer.fit_transform(train_data_numeric)

train_data_numeric = pd.DataFrame(train_data_numeric, columns = train_data_numeric_columns)
def df_diff(first, second):

        second = set(second)

        return [item for item in first if item not in second]
train_data_non_numeric = train_data1[df_diff(train_data1.columns, train_data_numeric.columns)]

train_data_non_numeric = pd.get_dummies(train_data_non_numeric)
train_data_preprocessed = pd.concat([train_data_numeric.reset_index(drop=True), train_data_non_numeric], axis=1)
test_data_numeric = test_data1.select_dtypes(exclude=['object'])

test_data_numeric_columns = test_data_numeric.columns

my_imputer = SimpleImputer()

test_data_numeric = my_imputer.fit_transform(test_data_numeric)

test_data_numeric = pd.DataFrame(test_data_numeric, columns = test_data_numeric_columns)
test_data_non_numeric = test_data[df_diff(test_data1.columns, test_data_numeric.columns)]

test_data_non_numeric = pd.get_dummies(test_data_non_numeric)
test_data_preprocessed = pd.concat([test_data_numeric.reset_index(drop=True), test_data_non_numeric], axis=1)
common_columns = test_data_preprocessed.columns.intersection(train_data_preprocessed.columns)

test_data_preprocessed = test_data_preprocessed[common_columns]

train_data_preprocessed = train_data_preprocessed[common_columns]
from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(train_data_preprocessed, train_data_y)

predictions_prices = xgb.predict(test_data_preprocessed)
predictions_prices
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions_prices})
my_submission.to_csv('sample_submission.csv', index=False)