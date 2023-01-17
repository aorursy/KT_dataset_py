import numpy as np

import pandas as pd



import lightgbm as lgb

from sklearn import model_selection
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train_features['TARGET'] = 1

test_features['TARGET'] = 0
data = pd.concat([train_features, test_features])
data['cp_type'].value_counts(normalize=True)
data = data[data['cp_type'] == 'trt_cp'].copy()

data.head()
data = data.drop(['sig_id', 'cp_type'], axis=1)
data['cp_dose'] = data['cp_dose'].astype('category')
X_data = data.drop(['TARGET'], axis=1)

y_data = data['TARGET']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_data, y_data, train_size=0.33, shuffle=True)
train = lgb.Dataset(X_train, label=y_train)

test = lgb.Dataset(X_test, label=y_test)
param = {'num_leaves': 50,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 5,

         'learning_rate': 0.2,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 44,

         "metric": 'auc',

         "verbosity": -1}
num_round = 50

clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)