import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')
data.head()
data = data[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']]

columns = data.columns

target_label = 'price'
float_columns = columns[(data.dtypes == 'float64').values]

categorical_columns = columns[(data.dtypes == 'object').values]

int_columns = columns[(data.dtypes == 'int64').values]
from sklearn.preprocessing import LabelEncoder
def oneHotEncoding(data, col):

    df = pd.get_dummies(data[col], prefix = col)

    data = pd.concat([data, df], axis = 1)

    data.drop([col], axis = 1, inplace = True)

    return data
def lableEncoder(data, col):

    data[col] = LabelEncoder().fit_transform(data[col])

    return data
encode = 'label'

for col in categorical_columns:

    if encode == 'label':

        data[col] = LabelEncoder().fit_transform(data[col])

    else:

        data = oneHotEncoding(data, col)
from sklearn.preprocessing import StandardScaler
for col in float_columns:

    data[col] = StandardScaler().fit_transform(np.array(data[col]).reshape(-1, 1))
sns.kdeplot(data['price'], label = 'price', shade = True)
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error
train, target = data.drop([target_label], axis = 1), data[target_label]

X_train, X_valid, Y_train, Y_valid = train_test_split(train, target, test_size = 0.3, random_state = 1)

columns = X_train.columns
sns.kdeplot(Y_train, shade = True, label = 'train')

sns.kdeplot(Y_valid, shade = True, label = 'validation')
def feature_importance(model, columns, model_name):

    df = pd.DataFrame(dict(zip(columns, model.feature_importances_.reshape(-1, 1)))).T

    df.columns = ['weights']

    df.plot(kind = 'bar', title = model_name)
param = {'n_estimators' : 100, 'n_jobs' : 10, 'learning_rate' : 0.1, 'max_depth' : 6}

xgb = XGBRegressor(**param)

xgb.fit(X_train, Y_train)
y_pred_valid = xgb.predict(X_valid)

y_pred_train = xgb.predict(X_train)

print('r2_score train: {} and validation: {}'.format(r2_score(y_pred_train, Y_train), r2_score(y_pred_valid, Y_valid)))

print('mean_absolute_error train: {} and validation: {}'.format(mean_absolute_error(y_pred_train, Y_train), mean_absolute_error(y_pred_valid, Y_valid)))
feature_importance(xgb, columns, 'XGBRegressor')
catboost = CatBoostRegressor()

catboost.fit(X_train, Y_train)
y_pred_valid = catboost.predict(X_valid)

y_pred_train = catboost.predict(X_train)

print('r2_score train: {} and validation: {}'.format(r2_score(y_pred_train, Y_train), r2_score(y_pred_valid, Y_valid)))

print('mean_absolute_error train: {} and validation: {}'.format(mean_absolute_error(y_pred_train, Y_train), mean_absolute_error(y_pred_valid, Y_valid)))
feature_importance(catboost, columns, 'CatBoostRegressor')
d_tree = DecisionTreeRegressor()

d_tree.fit(X_train, Y_train)
y_pred_valid = d_tree.predict(X_valid)

y_pred_train = d_tree.predict(X_train)

print('r2_score train: {} and validation: {}'.format(r2_score(y_pred_train, Y_train), r2_score(y_pred_valid, Y_valid)))

print('mean_absolute_error train: {} and validation: {}'.format(mean_absolute_error(y_pred_train, Y_train), mean_absolute_error(y_pred_valid, Y_valid)))
feature_importance(d_tree, columns, 'DecisionTreeRegressor')