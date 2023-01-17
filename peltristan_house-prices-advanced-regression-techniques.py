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
#Load necessary modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
raw_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
raw_test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

raw_data_no_na = raw_data.dropna(subset = ['SalePrice'],axis=0) #Drop rows with missing Target


y_train_full = raw_data_no_na['SalePrice'] #split X and y
X_train_full = raw_data_no_na.drop('SalePrice', axis=1)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, train_size=0.8, test_size=0.2, random_state=1337) #split to train and val sets

X_train = X_train[y_train < y_train.quantile(0.995)] #Drop the high price outliers
y_train = y_train[y_train < y_train.quantile(0.995)]

#Get numeric & categorical columns
cat_columns = [col for col in X_train.columns if
               X_train[col].dtype == 'object']


numeric_columns = [col for col in X_train.columns
                   if X_train[col].dtype == ('int64')]
#Take a look at the price historgramm
plt.hist(y_train[y_train.quantile(0.01) < y_train][y_train < y_train.quantile(0.99)])
plt.show()
print(y_train[y_train<50000].count())
numerical_transformer = SimpleImputer(strategy='median')

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessing = ColumnTransformer(
    transformers =[
        ('num', numerical_transformer, numeric_columns),
        ('cat', cat_transformer, cat_columns)
    ])

model = XGBRegressor(n_estimators=1000, learning_rate=0.02, n_jobs=4)

my_transformer = Pipeline(steps=[
    ('prep', preprocessing),
    ('model', model)
    ])
X_train_pre = preprocessing.fit_transform(X_train)
X_val_pre = preprocessing.transform(X_val)

my_transformer.fit(X_train, y_train, 
             model__early_stopping_rounds=5, 
             model__eval_set=[(X_val_pre, y_val)], 
             )
pred_val = my_transformer.predict(X_val)
print(mean_absolute_error(pred_val, y_val))
preds_test = my_transformer.predict(X_test_full)
#Same preprocessing, Random Forest model
model2 = RandomForestRegressor(criterion='mse',
                              n_estimators=1750,
                              min_samples_split=6,
                              min_samples_leaf=6,
                              max_features='auto',
                              random_state=1337,
                              n_jobs=-1,
                              verbose=1)

my_transformer2 = Pipeline(steps=[
    ('prep', preprocessing),
    ('model', model2)
    ])
my_transformer2.fit(X_train, y_train)
pred_val2 = my_transformer2.predict(X_val)
print(mean_absolute_error(pred_val2, y_val))
iterative_transformer = IterativeImputer(verbose=False, random_state=1337)

iter_cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),#IterativeImputer(verbose=False, random_state=1337, initial_strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

iter_preprocessing = ColumnTransformer(
    transformers =[
        ('num', iterative_transformer, numeric_columns),
        ('cat', iter_cat_transformer, cat_columns)
    ])

iter_my_transformer = Pipeline(steps=[
    ('prep', iter_preprocessing),
    ('model', model)
    ])
iter_preprocessing.fit_transform(X_train)
X_val_iter_pre = iter_preprocessing.transform(X_val)
iter_my_transformer.fit(X_train, y_train, 
             model__early_stopping_rounds=5, 
             model__eval_set=[(X_val_iter_pre, y_val)], 
             )
iter_pred_val = iter_my_transformer.predict(X_val)
print(mean_absolute_error(iter_pred_val, y_val))
preds_test = iter_my_transformer.predict(X_test_full)
output = pd.DataFrame({'Id': X_test_full['Id'],
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
# Save test predictions to file
output = pd.DataFrame({'Id': X_test_full['Id'],
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)