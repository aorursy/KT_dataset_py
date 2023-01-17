import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import mean_squared_error

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
train_cat = train.select_dtypes(include = 'object').drop(['Id'], axis = 1)

train_num = train.select_dtypes(include = ['float64', 'int64'])

train_cat_dummies = pd.get_dummies(train_cat)

train = pd.concat([train_num, train_cat_dummies], axis = 1)
train = train.fillna(train.mean())
train.info()
train.describe()
train.corr()['Target'].sort_values(ascending = False)
sample = train.sample(frac = .5)

y = sample['Target']

X = sample.drop(['Target'], axis = 1)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 99)

x_tr, x_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = .2, random_state = 99)
param_dict = {"n_estimators": [100]}

reg = RandomForestRegressor(n_jobs = -1, max_depth = 5)

gs = GridSearchCV(reg, param_dict, scoring = 'neg_mean_squared_error', cv = 2)

gs.fit(x_tr, y_tr)
train_predictions = gs.predict(x_tr)

train_error = mean_squared_error(y_tr, train_predictions)

val_predictions = gs.predict(x_val)

val_error = mean_squared_error(y_val, val_predictions)



print(train_error)

print(val_error)
feat_imports = sorted(list(zip(X_train.columns, gs.best_estimator_.feature_importances_)), key=lambda x:x[1], reverse=True)

feat_imports
reg_final = RandomForestRegressor(n_jobs = -1, max_depth = 3, n_estimators = 1000)

reg_final.fit(X_train, y_train)
test_predictions = reg_final.predict(X_test)

test_error = mean_squared_error(y_test, test_predictions)