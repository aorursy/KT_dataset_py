# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head(5)
train_df.columns.values

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df["target"]
train_df.drop(['ID_code', 'target'], inplace=True, axis=1)
# ada = SMOTE(random_state=42)

# train_df, target = ada.fit_resample(train_df, target)
x = train_df.values

y=target.values

x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)



train_data = lightgbm.Dataset(x, label=y)

test_data = lightgbm.Dataset(x_test, label=y_test)
parameters  = {

    'bagging_freq': 5,  'bagging_fraction': 0.331,  'boost_from_average':'false',   

    'boost': 'gbdt',    'feature_fraction': 0.0405, 'learning_rate': 0.0083,

    'max_depth': -1,    'metric':'auc',             'min_data_in_leaf': 400,     

    'min_sum_hessian_in_leaf': 10.0,'num_leaves': 13,  'num_threads': 8,            

    'tree_learner': 'serial',   'objective': 'binary',       'verbosity': 1

}

model = lightgbm.train(parameters,train_data,valid_sets=test_data,num_boost_round=6000,early_stopping_rounds=5000)
submission = pd.read_csv('../input/test.csv')

ids = submission['ID_code'].values

submission.drop('ID_code', inplace=True, axis=1)





x = submission.values

y = model.predict(x)



output = pd.DataFrame({'ID_code': ids, 'target': y})

output.to_csv("submission.csv", index=False)