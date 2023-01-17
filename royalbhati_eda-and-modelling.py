# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import warnings
warnings.filterwarnings("ignore")
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns

train =  pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape ,test.shape
train.head()
test.head()
train.dtypes
train.describe()
train['groupId'].nunique()
train.isnull().any()
test.isnull().any()
sns.distplot(train['winPlacePerc'],bins=100)
train['groupId'].nunique()
train['Id'].nunique()
test['groupId'].nunique()
train['matchId'].nunique()
sns.distplot(train['damageDealt'],bins=10)
sns.distplot(np.log1p(train['damageDealt']),bins=10)

sns.distplot(train['killPlace'],bins=100)
sns.distplot(train['killPoints'],bins=10)
sns.boxplot(train['killPoints'])
sns.violinplot(train['killPoints'])
sns.distplot(train['kills'],bins=10)
sns.boxplot(train['kills'])
sns.distplot(train['winPoints'])
sns.boxplot(train['winPoints'])



X=train.drop(['Id','winPlacePerc'],axis=1)
Y=train['winPlacePerc']
te=test.drop(['Id'],axis=1)
from sklearn.model_selection import train_test_split
dev_X, val_X, dev_y, val_y = train_test_split(X, Y, test_size = 0.1, random_state = 42)
# from sklearn.metrics import roc_auc_score
# from catboost import CatBoostRegressor
# # categorical_features_indices = [0,1,2,3,4,5,6,7,8,9]

# model=CatBoostRegressor(iterations=100, depth=12, learning_rate=0.09, loss_function= 'MAE')
# model.fit(dev_X,dev_y,plot=True)


import lightgbm as lgb 
import xgboost as xgb

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
    'learning_rate': 0.1,
    'max_depth': -1,
    'num_leaves': 40,
    'feature_fraction': 0.6,
    'min_data_in_leaf': 100,
    'lambda_l2': 4,
    'objective': 'regression_l2', 
    'metric': 'mae',
    'seed': 42}

    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=150, 
                      evals_result=evals_result)
    
    pred_test_y =model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y,te)
print("LightGBM Training Completed...")
test['winPlacePerc'] = pred_test
submission = test[['Id', 'winPlacePerc']]
submission.to_csv('submission_lgb.csv', index=False)
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:50])
