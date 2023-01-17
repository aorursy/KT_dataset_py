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
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.shape)
train.head()
print(test.shape)
test.head()
train.describe()
na_train_columns=train.columns[train.isna().any()]
print("Missing values for train set")
print(train[na_train_columns].isna().sum())

print('---------------------------------------------------')
na_test_columns=test.columns[test.isna().any()]
print("Missing values for test set")
print(test[na_test_columns].isna().sum())
train = train.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)
test = test.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)
for column_name in train.columns:
    if train[column_name].dtypes == 'object':
        train[column_name] = train[column_name].fillna(train[column_name].mode().iloc[0])
    elif train[column_name].dtypes == 'int64' or 'float64':
        train[column_name] = train[column_name].fillna(train[column_name].mean())
    else:
        continue
        
for column_name in test.columns:
    if test[column_name].dtypes == 'object':
        test[column_name] = test[column_name].fillna(test[column_name].mode().iloc[0])
    elif test[column_name].dtypes == 'int64' or 'float64':
        test[column_name] = test[column_name].fillna(test[column_name].mean())
    else:
        continue
na_train_columns2 = train.columns[train.isna().any()]
print("Missing values for train set")
print(train[na_train_columns2].isna().sum())

print('---------------------------------------------------')
na_test_columns2 = test.columns[test.isna().any()]
print("Missing values for test set")
print(test[na_test_columns2].isna().sum())
y = train['SalePrice']
train = train.drop(["SalePrice"], axis = 1)
ID = test['Id']
train = train.drop("Id", axis=1)
test = test.drop("Id", axis=1)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.shape)
print(test.shape)
missing_cols = set( train.columns ) - set( test.columns )
print( missing_cols)
train = train.drop(missing_cols, axis=1)
print(train.shape)
print(test.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size = 0.10, random_state=42)
from sklearn import ensemble

# Gradient Boosting Regressor
gbr = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, 
                                         learning_rate=0.1, loss='ls' )
gbr.fit(x_train, y_train)
print(("Prediction score for GradientBoostingRegressor: {}").format(gbr.score(x_test, y_test)))

# Random Forest Regressor
rfg = ensemble.RandomForestRegressor(n_estimators=400)
rfg.fit(x_train, y_train)
print(("Prediction score for RandomForestRegressor: {}").format(rfg.score(x_test, y_test)))

# Extra Tree Regressor
etg = ensemble.ExtraTreesRegressor(n_estimators=400 , max_depth = 5)
etg.fit(x_train, y_train)
print(("Prediction score for ExtraTreeRegressor: {}").format(etg.score(x_test, y_test)))
# place columns in both train and test set in the same order
test = test[train.columns]
pred = gbr.predict(test)
submission = pd.DataFrame()
submission["Id"] = ID
submission["SalePrice"] = pred.round(2)

submission.to_csv("my_submission.csv", index=False)
submission.head(10)