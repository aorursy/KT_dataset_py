import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
print('Loading dataset...')
sample_size = None
X = pd.read_csv('../input/creditcard.csv', nrows=sample_size)
#Plot time distributions
sns.distplot(X[X['Class']==0]['Time'])
sns.distplot(X[X['Class']==1]['Time'])
#Feature engineering
#Momento of the day
X['Time'] = X['Time'] % (24*60*60)
y = X.pop('Class')
# create dataset for lightgbm
lgbm_train = lgbm.Dataset(X, y)

# specify your configurations as a dict
lgbm_params = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.02,
    'min_data_in_leaf': 30,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.15,
    'scale_pos_weight': 2,
    'drop_rate': 0.02
}

print('Start CV...')
cv_results = lgbm.cv(train_set=lgbm_train,
                     params=lgbm_params,
                     nfold=6,
                     show_stdv=True,
                     num_boost_round=1000000,
                     early_stopping_rounds=500,
                     stratified=True,
                     verbose_eval=50,
                     metrics=['auc'])

print('Best CV result = {}'.format(np.max(cv_results['auc-mean'])))