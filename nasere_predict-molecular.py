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
import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn import metrics

from sklearn import linear_model

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/train.csv") # readingthe training set

test = pd.read_csv("../input/test.csv") # reading the test set

sub = pd.read_csv("../input/sample_submission.csv")

structures = pd.read_csv("../input/structures.csv")
train.head()
test.head()
print(f'There are {train.shape[0]} rows in train data.')

print(f'There are {test.shape[0]} rows in test data.')



print(f"There are {train['molecule_name'].nunique()} distinct molecules in train data.")

print(f"There are {test['molecule_name'].nunique()} distinct molecules in test data.")

print(f"There are {train['atom_index_0'].nunique()} unique atoms.")

print(f"There are {train['type'].nunique()} unique types.")
len(structures)




def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)



train.head()
train.describe()
# checking some variable unique couunt

print(train['atom_index_0'].nunique())

print(train['atom_index_1'].nunique())

print(train['id'].nunique())
train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

train['dist_x'] = (train['x_0'] - train['x_1']) ** 2

test['dist_x'] = (test['x_0'] - test['x_1']) ** 2

train['dist_y'] = (train['y_0'] - train['y_1']) ** 2

test['dist_y'] = (test['y_0'] - test['y_1']) ** 2

train['dist_z'] = (train['z_0'] - train['z_1']) ** 2

test['dist_z'] = (test['z_0'] - test['z_1']) ** 2
train.isnull().sum() # no missing value
# features from groupby

train['molecule_name_unique'] = train['molecule_name'].map(train.groupby(train['molecule_name'])['molecule_name'].nunique())

test['molecule_name_unique'] = test['molecule_name'].map(test.groupby(test['molecule_name'])['molecule_name'].nunique())

train['molecule_name_type'] = train['molecule_name'].map(train.groupby(train['molecule_name'])['type'].nunique())

test['molecule_name_type'] = test['molecule_name'].map(test.groupby(test['molecule_name'])['type'].nunique())

train['molecule_dist_mean'] = train['molecule_name'].map(train.groupby(train['molecule_name'])['dist'].mean())

test['molecule_dist_mean'] = test['molecule_name'].map(test.groupby(test['molecule_name'])['dist'].mean())

train['molecule_dist_sum'] = train['molecule_name'].map(train.groupby(train['molecule_name'])['dist'].sum())

test['molecule_dist_sum'] = test['molecule_name'].map(test.groupby(test['molecule_name'])['dist'].sum())

train['molecule_dist_min'] = train['molecule_name'].map(train.groupby(train['molecule_name'])['dist'].min())

test['molecule_dist_min'] = test['molecule_name'].map(test.groupby(test['molecule_name'])['dist'].min())

train['molecule_atom_count'] = train['molecule_name'].map(train.groupby(train['molecule_name'])['atom_1'].count())

test['molecule_atom_count'] = test['molecule_name'].map(test.groupby(test['molecule_name'])['atom_1'].count())

train['molecule_atom_u'] = train['molecule_name'].map(train.groupby(train['molecule_name'])['atom_1'].nunique())

test['molecule_atom_u'] = test['molecule_name'].map(test.groupby(test['molecule_name'])['atom_1'].nunique())

# by type

train['type_unique'] = train['type'].map(train.groupby(train['type'])['type'].nunique())

test['type_unique'] = test['type'].map(test.groupby(test['type'])['type'].nunique())

train['type_dist_mean'] = train['type'].map(train.groupby(train['type'])['dist'].mean())

test['type_dist_mean'] = test['type'].map(test.groupby(test['type'])['dist'].mean())

train['type_dist_sum'] = train['type'].map(train.groupby(train['type'])['dist'].sum())

test['type_dist_sum'] = test['type'].map(test.groupby(test['type'])['dist'].sum())

train['type_dist_min'] = train['type'].map(train.groupby(train['type'])['dist'].min())

test['type_dist_min'] = test['type'].map(test.groupby(test['type'])['dist'].min())

train['type_atom_count'] = train['type'].map(train.groupby(train['type'])['atom_1'].count())

test['type_atom_count'] = test['type'].map(test.groupby(test['type'])['atom_1'].count())

train['type_atom_u'] = train['type'].map(train.groupby(train['type'])['atom_1'].nunique())

test['type_atom_u'] = test['type'].map(test.groupby(test['type'])['atom_1'].nunique())
train.head()
#train = train.drop(['id', 'type_atom_u'], axis=1)

#test = test.drop(['id', 'type_atom_u'], axis=1)
object_data = train.dtypes[train.dtypes == 'object'].index
train = train.drop(['atom_0', 'atom_1'], axis=1)

test = test.drop(['atom_0', 'atom_1'], axis=1)
object_data
train['molecule_name'] = train['molecule_name'].astype('category').cat.codes

test['molecule_name'] = test['molecule_name'].astype('category').cat.codes
# dummies the remaing 

train = pd.get_dummies(train)

test = pd.get_dummies(test)
train.head()
X = train.drop('scalar_coupling_constant', axis=1)

y = train['scalar_coupling_constant']

X_test = test
n_fold = 3

folds = KFold(n_splits=n_fold, shuffle=True, random_state=100)
from xgboost import XGBRegressor 
params = {#'num_leaves': 12,

          'min_child_samples': 3,

          'objective': 'regression',

          'max_depth': 7,

          'learning_rate': 0.1,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.9,

          #"bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.9,

          'reg_lambda': 0.9,

          'colsample_bytree': 0.8

         }

result_dict_lgb = XGBRegressor(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='xgb', eval_metric='group_mae', plot_feature_importance=True,

                                                      verbose=500, early_stopping_rounds=100, n_estimators=1000)



model = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=500, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(test)
sub['scalar_coupling_constant'] = pred

sub.to_csv('chemistry.csv', index=False)

sub.head()