import numpy as np

import pandas as pd

from datetime import datetime as dt

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Imputer, StandardScaler

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



import os

print(os.listdir("../input"))
ss = pd.read_csv('../input/sample_submission.csv')

ss.head()
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_total = pd.concat([df_train, df_test])

print(df_train.shape, df_test.shape, df_total.shape)

df_total.head()
cols_train = set(df_train.columns)

cols_test = set(df_test.columns)

print(cols_train - cols_test)

print(cols_test - cols_train)
def count_nans(df):

    return df.isnull().sum().sum()



count_nans(df_train), count_nans(df_test)
target = df_train['SalePrice']

df_total.drop(columns=['Id', 'SalePrice'], inplace=True)

n_train = df_train.shape[0]
df_numeric = df_total.select_dtypes(include='number')

cols_numeric = list(df_numeric.columns)

cols_categoric = [col for col in df_total.columns if col not in cols_numeric]

df_onehot = pd.get_dummies(df_total)

print(df_onehot.shape)
# pipe = Pipeline([

#     ('imputer', Imputer(strategy='median')),

#     ('scaler', StandardScaler()),

#     ('regressor', Ridge(random_state=42))

# ])

# params = {'regressor__alpha': [10**i for i in range(-3, 4)]}

# print(params)

# grid_search = GridSearchCV(pipe, params, scoring='neg_mean_squared_error', n_jobs=-1, cv=10)

# grid_search.fit(df_numeric[:n_train], target)

# print(grid_search.best_score_)

# print(grid_search.best_params_)



model = Pipeline([

    ('imputer', Imputer(strategy='median')),

#     ('regressor', LGBMRegressor())    

    ('regressor', RandomForestRegressor())

])

model.fit(df_onehot[:n_train], target)
id = df_test['Id']

test_data = df_onehot[n_train:]

# y_pred = grid_search.best_estimator_.predict(test_data)

y_pred = model.predict(test_data)

df_ans = pd.DataFrame({'Id': id, 'SalePrice': y_pred})

now = dt.now()

fname = 'subm_{}_{}__{}_{}.csv'.format(now.day, now.month, now.hour, now.minute)

df_ans.to_csv(fname, index=False)