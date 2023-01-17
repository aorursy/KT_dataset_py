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
data = pd.read_csv('../input/house-price-prediction-challenge/train.csv')
X_test= pd.read_csv('../input/house-price-prediction-challenge/test.csv')
data.shape, X_test.shape
X.dtypes
target = 'TARGET(PRICE_IN_LACS)'
X = data.dropna(axis=0, subset=[target])
y = data[target]
X.drop([target], axis=1, inplace=True)
X.shape
X.isnull().sum()
X.nunique()
categorical_var = X.select_dtypes(include='object').columns
categorical_var
#Approach 1: dropping categorical varibles
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return r2_score(y_valid, preds)

from xgboost import XGBRegressor
def score_dataset_x(X_train, X_valid, y_train, y_valid):
    model = XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return r2_score(y_valid, preds)

reduced_X_train = X_train.drop(categorical_var, axis=1)
reduced_X_valid = X_valid.drop(categorical_var, axis=1)

print("MSLE from Approach 1 (drop cat cols):")
print('RF: ',score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)) 
print('XGB: ',score_dataset_x(reduced_X_train, reduced_X_valid, y_train, y_valid)) 

model = XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=0, n_jobs=3)
model.fit(pd.concat([reduced_X_train, reduced_X_valid], axis=0), pd.concat([y_train, y_valid], axis=0))
reduced_X_test = X_test.drop(categorical_var, axis=1)
preds_test = model.predict(reduced_X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
print('Done!')


