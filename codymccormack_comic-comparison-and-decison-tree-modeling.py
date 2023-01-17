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
marvel_data = pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv', index_col='page_id')

dc_data = pd.read_csv('../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv', index_col='page_id')
marvel_data.head()
dc_data.head()
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))

sns.barplot(x=marvel_data['SEX'], y=marvel_data['APPEARANCES'])
plt.figure(figsize=(20,10))

sns.barplot(x=dc_data['SEX'], y=dc_data['APPEARANCES'])
test = marvel_data[marvel_data['SEX'] == 'Genderfluid Characters']

test.head()
from datetime import datetime

marvel_data = marvel_data.dropna(subset=['Year'])

marvel_data['Year'] = pd.to_datetime(marvel_data['Year'], format='%Y')

marvel_data['Year'] = marvel_data['Year'].dt.year

marvel_data['AVERAGE APPEARANCES PER YEAR'] = marvel_data['APPEARANCES'] / (datetime.today().year - marvel_data['Year'])
plt.figure(figsize=(20,10))

sns.barplot(x=marvel_data['SEX'], y=marvel_data['AVERAGE APPEARANCES PER YEAR'])
marvel_data = marvel_data.drop(index=[2042, 23853], axis=0)

plt.figure(figsize=(20,10))

sns.barplot(x=marvel_data['SEX'], y=marvel_data['AVERAGE APPEARANCES PER YEAR'])
dc_data = dc_data.dropna(subset=['YEAR'])

dc_data['YEAR'] = pd.to_datetime(dc_data['YEAR'], format='%Y')

dc_data['YEAR'] = dc_data['YEAR'].dt.year

dc_data['AVERAGE APPEARANCES PER YEAR'] = dc_data['APPEARANCES'] / (datetime.today().year - dc_data['YEAR'])
plt.figure(figsize=(20,10))

sns.barplot(x=dc_data['SEX'], y=dc_data['AVERAGE APPEARANCES PER YEAR'])
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

marvel_data = marvel_data.dropna(subset=['EYE', 'HAIR', 'SEX', 'GSM', 'AVERAGE APPEARANCES PER YEAR'])

marvel_data.EYE = le.fit_transform(marvel_data.EYE)

marvel_data.HAIR = le.fit_transform(marvel_data.HAIR)

marvel_data.SEX = le.fit_transform(marvel_data.SEX)

marvel_data.GSM = le.fit_transform(marvel_data.GSM)

marvel_features = ['EYE', 'HAIR', 'SEX', 'GSM']

y = marvel_data['AVERAGE APPEARANCES PER YEAR']

X = marvel_data[marvel_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

from xgboost import XGBRegressor

marvel_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=5, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.300000012, max_delta_step=0, max_depth=3,

             min_child_weight=1, monotone_constraints='()',

             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1.0,

             tree_method='exact', validate_parameters=1, verbosity=None)

marvel_model.fit(train_X, train_y)

predictions = marvel_model.predict(val_X)

mean_absolute_error(val_y, predictions)

#4.984222164541463 MAE with decision tree

#4.040619781347345 MAE with XGboost

#5.893323851698426 MAE with base regressor

#4.00539439707728 with optimized regressor
from sklearn.model_selection import GridSearchCV



params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }



grid = GridSearchCV(marvel_model, param_grid=params, n_jobs=4, cv=5, verbose=3)

grid.fit(X, y)

print('\n Best estimator:')

print(grid.best_estimator_)
dc_data = dc_data.dropna(subset=['EYE', 'HAIR', 'SEX', 'GSM', 'AVERAGE APPEARANCES PER YEAR'])

dc_data.EYE = le.fit_transform(dc_data.EYE)

dc_data.HAIR = le.fit_transform(dc_data.HAIR)

dc_data.SEX = le.fit_transform(dc_data.SEX)

dc_data.GSM = le.fit_transform(dc_data.GSM)

dc_features = ['EYE', 'HAIR', 'SEX', 'GSM']

y = dc_data['AVERAGE APPEARANCES PER YEAR']

X = dc_data[dc_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

dc_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.6, gamma=5, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.300000012, max_delta_step=0, max_depth=3,

             min_child_weight=5, monotone_constraints='()',

             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1.0,

             tree_method='exact', validate_parameters=1, verbosity=None)

dc_model.fit(train_X, train_y)

predictions = dc_model.predict(val_X)

mean_absolute_error(val_y, predictions)

#1.5486301984506026 MAE with decision tree

#1.1755523030016262 with XGboost

#1.3411780180951278 with base regressor

#1.7037177049288135 MAE with optimized regressor =()
'''from sklearn.model_selection import GridSearchCV



params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }



grid = GridSearchCV(dc_model, param_grid=params, n_jobs=4, cv=5, verbose=3)

grid.fit(X, y)

print('\n Best estimator:')

print(grid.best_estimator_)'''