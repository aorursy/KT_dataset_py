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
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv' , index_col = 'Id')
train_df
train_df.isna()
train_df.isna().sum()
cols_with_missing = [col for col in train_df.columns
                     if train_df[col].isnull().any()]
print(cols_with_missing)
for col in cols_with_missing:
    print (train_df[col].isnull().sum())
cols_with_big_missing = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
train_df_after =  train_df.drop(cols_with_big_missing, axis=1)
train_df_after
train_df_after.dropna(axis=0, inplace=True)
train_df_after
s = (train_df_after.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
from sklearn.preprocessing import LabelEncoder
label_train_df_after = train_df_after.copy()
label_encoder = LabelEncoder()
for col in object_cols:
    label_train_df_after[col] = label_encoder.fit_transform(train_df_after[col])
target_col = 'SalePrice'
y = label_train_df_after[target_col]
X = label_train_df_after.drop(columns=[target_col])
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
best_para = max(results, key=results.get)
print('best para', best_para)
print('value', results[best_para])
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
test_df
cols_with_missing_test = [col for col in test_df.columns
                     if test_df[col].isnull().any()]
cols_with_missing_test
for col in cols_with_missing_test:
    print (test_df[col].isnull().sum())
cols_with_big_missing = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
test_data = test_df.drop(cols_with_big_missing, axis=1)

cols_with_missing_test = [col for col in test_data.columns
                     if test_data[col].isnull().any()]


for col in cols_with_missing_test:
    print (test_data[col].isnull().sum())
s = (test_data.dtypes == 'object')
object_cols_test = list(s[s].index)

print("Categorical variables:")
print(object_cols_test)

final_model = RandomForestRegressor(n_estimators=best_para)
final_model.fit(X, y)
#predictions = final_model.predict(label_test_data)
#predictions[:5]