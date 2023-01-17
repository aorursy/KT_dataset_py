# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/flight_delays_train.csv')
train.head(10)
train['DayOfWeek'] = train.DayOfWeek.apply(lambda x: int(x.split('-')[1]))
train['Month'] = train.Month.apply(lambda x: int(x.split('-')[1]))
train['DayofMonth'] = train.DayofMonth.apply(lambda x: int(x.split('-')[1]))

train['dep_delayed_15min'] = train.dep_delayed_15min.apply(lambda x: 1 if x == 'Y' else 0)

origin_list = train.Origin.value_counts().head(10).index.tolist()
train['Origin'] = train.Origin.apply(lambda x: x if x in origin_list else 'other')

dest_list = train.Dest.value_counts().head(10).index.tolist()
train['Dest'] = train.Dest.apply(lambda x: x if x in dest_list else 'other')

carriers_list = train.UniqueCarrier.value_counts().head(10).index.tolist()
train['UniqueCarrier'] = train.UniqueCarrier.apply(lambda x: x if x in carriers_list else 'other')

train = pd.get_dummies(train)
X_train, X_test, y_train, y_test = train_test_split(train.drop(['dep_delayed_15min'],axis=1), train.dep_delayed_15min, 
                                                    test_size=0.33, random_state=42, stratify=train.dep_delayed_15min)

clf = RandomForestClassifier(n_estimators=100, max_depth=10)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
gbm = lgb.LGBMClassifier(n_estimators=200,)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary',
          'nthread': 3,
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'}

# Create parameters to search
gridParams = {
    'learning_rate': [0.5, 0.005],
    'n_estimators': [40, 200, 1000],
    'num_leaves': [5, 10, 15],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501], 
    'colsample_bytree' : [0.66],
    'subsample' : [0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.4],
    }

# Create classifier to use
mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'auc',
          n_jobs = 3, 
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])
grid = GridSearchCV(mdl, gridParams,
                    verbose=3,
                    cv=4,
                    n_jobs=2)
# Run the grid
grid.fit(train.drop('dep_delayed_15min', axis=1), train.dep_delayed_15min)
grid.best_params_
gbm = lgb.LGBMClassifier(**grid.best_params_)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)
gbm = lgb.LGBMClassifier(**grid.best_params_)
gbm.fit(train.drop('dep_delayed_15min', axis=1), train.dep_delayed_15min)
roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1])
test = pd.read_csv('../input/flight_delays_test.csv')
test['DayOfWeek'] = test.DayOfWeek.apply(lambda x: int(x.split('-')[1]))
test['Month'] = test.Month.apply(lambda x: int(x.split('-')[1]))
test['DayofMonth'] = test.DayofMonth.apply(lambda x: int(x.split('-')[1]))

test['Origin'] = test.Origin.apply(lambda x: x if x in origin_list else 'other')
test['Dest'] = test.Dest.apply(lambda x: x if x in dest_list else 'other')
test['UniqueCarrier'] = test.UniqueCarrier.apply(lambda x: x if x in carriers_list else 'other')

test = pd.get_dummies(test)
my_submission = pd.DataFrame({'id': test.index, 'dep_delayed_15min': gbm.predict_proba(test)[:, 1]})
# you could use any filename. We choose submission here
my_submission.to_csv('submission3.csv', index=False)

