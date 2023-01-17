# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score

from sklearn.model_selection import KFold, StratifiedKFold

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
basic_path = '../input/feature-extraction/'

basic_path = '../input/data-preprocessing/'

train_path = 'treemodel_train.csv'

test_path = 'treemodel_test.csv'
train_data = pd.read_csv(basic_path + train_path)

test_data = pd.read_csv(basic_path + test_path)

train_data.head()
len(train_data.columns)
#train_data['label'] = train_data['label'].apply(lambda x:x if x<1 else 1)

train_data['label'] += 1

train_data['label'].hist()
num_folds = 5

folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=2019) #

oof_preds = np.zeros(train_data.shape[0])

sub_preds_ = []

sub_preds = np.zeros(test_data.shape[0])

sub_preds_ = np.zeros((test_data.shape[0],33))

feature_importance_df = pd.DataFrame()

feats = [f for f in train_data.columns if f not in ['user_id','listing_id','label']]

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_data[feats], train_data['label'])):

    train_x, train_y = train_data[feats].iloc[train_idx], train_data['label'].iloc[train_idx]

    valid_x, valid_y = train_data[feats].iloc[valid_idx], train_data['label'].iloc[valid_idx]    

    # LightGBM parameters found by Bayesian optimization

    clf = LGBMClassifier(

        nthread=4,

        n_estimators=1000,

        learning_rate=0.05,

        num_leaves=34,

        colsample_bytree=0.9497036,

        subsample=0.8715623,

        max_depth=8,

        reg_alpha=0.041545473,

        reg_lambda=0.0735294,

        min_split_gain=0.0222415,

        min_child_weight=39.3259775,

        silent=-1,

        verbose=100, )

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

        eval_metric= 'logloss', verbose= -1, early_stopping_rounds= 100)

    oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)

    sub_preds += clf.predict(test_data[feats], num_iteration=clf.best_iteration_)/ folds.n_splits

    sub_preds_ += clf.predict_proba(test_data[feats], num_iteration=clf.best_iteration_)/ folds.n_splits

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = feats

    fold_importance_df["importance"] = clf.feature_importances_

    fold_importance_df["fold"] = n_fold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #print('Fold %2d ACC : %.6f' % (n_fold + 1,accuracy_score(valid_y, oof_preds[valid_idx])))

    print('Fold %2d ACC : %.6f' % (n_fold + 1,accuracy_score(valid_y, oof_preds[valid_idx])))

feature_importance_df.to_csv('feature_importance.csv',index=False)
submsission = pd.DataFrame(sub_preds)

submsission['user_id'] = test_data['user_id']

submsission.to_csv('submsission.csv',index=False)
submsission_prob = pd.DataFrame(sub_preds_)

submsission_prob['user_id'] = test_data['user_id']

submsission_prob.to_csv('prob_submsission.csv',index=False)