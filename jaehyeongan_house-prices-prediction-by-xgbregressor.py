# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# laod data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.head()
# set index

train_df.set_index('Id', inplace=True)

test_df.set_index('Id', inplace=True)

len_train_df = len(train_df)

len_test_df = len(test_df)
corrmat = train_df.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>=0.3]

plt.figure(figsize=(13,10))

g = sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# split y_label

train_y_label = train_df['SalePrice']

train_df.drop(['SalePrice'], axis=1, inplace=True)
# concat train & test

boston_df = pd.concat((train_df, test_df), axis=0)

boston_df_index = boston_df.index

print(len(boston_df))

boston_df.head()
# check null 

check_null = boston_df.isna().sum() / len(boston_df)



# columns of null ratio >= 0.5

check_null[check_null >= 0.5]
# remove columns of null ratio >= 0.5

remove_cols = check_null[check_null >= 0.5].keys()

boston_df = boston_df.drop(remove_cols, axis=1)



boston_df.head()
# split object & numeric

boston_obj_df = boston_df.select_dtypes(include='object')

boston_num_df = boston_df.select_dtypes(exclude='object')
print('Object type columns:\n',boston_obj_df.columns)

print('Numeric type columns:\n',boston_num_df.columns)
boston_dummy_df = pd.get_dummies(boston_obj_df, drop_first=True)

boston_dummy_df.index = boston_df_index

boston_dummy_df.head()
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='mean')

imputer.fit(boston_num_df)

boston_num_df_ = imputer.transform(boston_num_df)
boston_num_df = pd.DataFrame(boston_num_df_, columns=boston_num_df.columns, index=boston_df_index)

boston_num_df.head()
boston_df = pd.merge(boston_dummy_df, boston_num_df, left_index=True, right_index=True)

print(len(boston_df))

boston_df.head()
train_df = boston_df[:len_train_df]

test_df = boston_df[len_train_df:]

print('train set length: ',len(train_df))

print('test set length: ',len(test_df))
train_df['SalePrice'] = train_y_label
from sklearn.model_selection import train_test_split

X_train = train_df.drop(['SalePrice'], axis=1)

y_train = train_df['SalePrice']



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)



X_test = test_df

test_id_idx = test_df.index
from sklearn.model_selection import GridSearchCV

import xgboost as xgb



param = {

    'max_depth':[3,4,5],

    'n_estimators':[250,300,330],

    'colsample_bytree':[0.3,0.5,0.7],

    'colsample_bylevel':[0.3,0.5,0.7],

    'gamma':[0.1,0,4,0.7]

}

model = xgb.XGBRegressor()



xgb_grid = GridSearchCV(estimator=model, param_grid=param, cv=5, 

                           scoring='neg_mean_squared_error',

                           n_jobs=-1)





xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_params_)

print(xgb_grid.best_estimator_)
xgb_grid.score(X_train, y_train)
test_y_pred = xgb_grid.predict(X_test)
id_pred_df = pd.DataFrame()

id_pred_df['Id'] = test_id_idx

id_pred_df['SalePrice'] = test_y_pred
id_pred_df.to_csv('submission.csv', index=False)
id_pred_df