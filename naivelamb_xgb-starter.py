# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



import xgboost as xgb



import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

random_state = 42

np.random.seed(random_state)
xgb_params = {

        'objective': 'binary:logistic',

        #'objective':'reg:linear',

        'tree_method': 'gpu_hist',

        'eta':0.1,

        'num_round':120000,

        'max_depth': 8,

        'silent':1,

        'subsample':0.5,

        'colsample_bytree': 0.5,

        'min_child_weight': 100,

        'eval_metric': 'auc',

        'verbose_eval': 1000,

    }
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

oof = df_train[['id', 'target']]

oof['predict'] = 0

predictions = df_test[['id']]

val_aucs = []

features = [col for col in df_train.columns if col not in ('id', 'target')]
for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):

    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']

    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']

    train_dataset = xgb.DMatrix(X_train, y_train)

    valid_dataset = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(train_dataset, 'train'), (valid_dataset, 'valid')]

    xgb_clf = xgb.train(xgb_params,

                        train_dataset,

                        evals=watchlist,

                        num_boost_round=12000,

                        early_stopping_rounds=300,

                        verbose_eval=1000

                       )

    p_valid = xgb_clf.predict(valid_dataset, ntree_limit=xgb_clf.best_iteration)

    yp = xgb_clf.predict(xgb.DMatrix(df_test[features]), ntree_limit=xgb_clf.best_iteration)

    

    oof['predict'][val_idx] = p_valid

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

    predictions['fold{}'.format(fold+1)] = yp
all_auc = roc_auc_score(oof['target'], oof['predict'])

print('ROC mean: %.6f, std: %.6f.' % (np.mean(val_aucs), np.std(val_aucs)))

print('Ensemble ROC: %.6f' % (all_auc))
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['id', 'target']]].values, axis=1)

sub["target"] = predictions['target']

sub.to_csv("submission.csv", index=False)

oof.to_csv('oof.csv', index=False)