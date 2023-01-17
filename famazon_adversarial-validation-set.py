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
# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

#Set labels for train/test data
train_df['TAR'] = 0
test_df['TAR'] = 1
# Get the combined data
total_df = pd.concat([train_df.drop('target', axis=1), test_df], axis=0).drop('ID', axis=1)
# Train and test
train_idx = range(0, len(train_df))
test_idx = range(len(train_df), len(total_df))
#Get labels
y = total_df.TAR.copy()
total_df.drop('TAR', axis = 1, inplace = True)
#Shuffle and split our set
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(total_df, y, test_size=0.20, shuffle = True, random_state = 42)
import lightgbm as lgb
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 32,
    'learning_rate': 0.02,
    'verbose': 0,
    'lambda_l1': 1,
    'scale_pos_weight': 8  #for unbalanced labels
} 
lgtrain = lgb.Dataset(X_train, y_train)

lgvalid = lgb.Dataset(X_valid, y_valid)

lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=10000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=100,
    verbose_eval=100
        )

train_preds = lgb_clf.predict(total_df.iloc[train_idx])
train_df['prob_to_test'] = train_preds
val_set = train_df[train_df.prob_to_test>0.9] #train set with prob more than 90% to be test set
val_set.head()
# feature importance
print("Features Importance...")
gain = lgb_clf.feature_importance('gain')
featureimp = pd.DataFrame({'feature':lgb_clf.feature_name(), 
                   'split':lgb_clf.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
featureimp.head()
