# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

taxi_data = pd.read_csv("../input/train.csv", nrows=200000)
taxi_data.head()
%%time
#correctly parse the date time column
data_update = taxi_data.copy()
s = pd.to_datetime(taxi_data.pickup_datetime)
print(0)
data_update['date_of_month'] = s.dt.day
print(1)
data_update['day_of_week'] = s.dt.dayofweek
print(2)
data_update['month'] = s.dt.month
print(3)
data_update['time'] = s.dt.hour + s.dt.minute/60
#data_update.time
to_train = data_update.select_dtypes(exclude=['object'])
y = to_train.fare_amount
X = to_train.drop(['fare_amount'], axis=1)
orig = X.copy()
to_train.head()
#impute the complete data
from sklearn.impute import SimpleImputer

new_data = X.copy()

# make new columns indicating what will be imputed
cols_with_missing = list(col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    print(col)
    new_data[col + '_was_missing'] = new_data[col].isnull()

my_imputer = SimpleImputer()
new_data_imputed = pd.DataFrame(my_imputer.fit_transform(new_data))
new_data_imputed.columns = new_data.columns
X = new_data_imputed
#split the data into training and test data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
%%time
from xgboost import XGBRegressor

nyc_model_xgb = XGBRegressor(n_estimators=4000, n_jobs=10)
#nyc_model_xgb.fit(train_X, train_y, early_stopping_rounds=10, 
#                   eval_set=[(val_X, val_y)], verbose=True)
nyc_model_xgb.fit(train_X, train_y)

predictions_xgb = nyc_model_xgb.predict(val_X)
xgb_val_mae = mean_absolute_error(predictions_xgb, val_y)

print(xgb_val_mae)
#load test file
test_data_path = '../input/test.csv'

test_data = pd.read_csv(test_data_path)
#test_X = test_data[orig.columns]
test_data.columns
#parse date time for the test data
test_update = test_data.copy()
s = pd.to_datetime(test_data.pickup_datetime)
print(0)
test_update['date_of_month'] = s.dt.day
print(1)
test_update['day_of_week'] = s.dt.dayofweek
print(2)
test_update['month'] = s.dt.month
print(3)
test_update['time'] = s.dt.hour + s.dt.minute/60
test_update.head()
#impute the test data file accordingly
new_test = test_update[orig.columns]
for col in cols_with_missing:
    print(col)
    new_test[col + '_was_missing'] = new_test[col].isnull()

#new_test.head()
# Imputation
#my_imputer = SimpleImputer()
new_test_imput = pd.DataFrame(my_imputer.fit_transform(new_test))
new_test_imput.columns = new_test.columns
test_X = new_test_imput
test_predictions = nyc_model_xgb.predict(test_X)
output = pd.DataFrame({'key': test_data.key,
                       'fare_amount': test_predictions})

output.to_csv('submission.csv', index=False)
