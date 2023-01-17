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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train['is_test'] = 0

test['is_test'] = 1

train = train.drop('target', axis=1)
df = pd.concat([train, test], axis=0, ignore_index=True)
df.head(1)
y = df['is_test']

X = df.drop(['is_test', 'id'], axis=1)
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
oof_preds1 = np.zeros(df.shape[0])

oof_preds2 = np.zeros(df.shape[0])

feats = list(X.columns)

feature_coef_df = pd.DataFrame()

feature_importance_df = pd.DataFrame()

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):

    train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]

    valid_X, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]

    

    clf1, clf2 = LogisticRegression(solver='liblinear'), LGBMClassifier(n_estimators=1000, max_depth=2)

    clf1.fit(train_X, train_y)

    clf2.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)], eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

    oof_preds1[valid_idx] = clf1.predict_proba(valid_X)[:, 1]

    oof_preds2[valid_idx] = clf2.predict_proba(valid_X)[:, 1]



    fold_coef_df = pd.DataFrame()

    fold_coef_df["feature"] = feats

    fold_coef_df["coef"] = clf1.coef_[0]

    fold_coef_df["fold"] = n_fold + 1

    feature_coef_df = pd.concat([feature_coef_df, fold_coef_df], axis=0)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = feats

    fold_importance_df["importance"] = clf2.feature_importances_

    fold_importance_df["fold"] = n_fold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    print('Fold %2d AUC LR : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds1[valid_idx])))    

    print('Fold %2d AUC LGBM : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds2[valid_idx])))



print('Full AUC score LR %.6f' % roc_auc_score(y, oof_preds1))

print('Full AUC score LGBM %.6f' % roc_auc_score(y, oof_preds2))
new_feature_coef_df = feature_coef_df

new_feature_coef_df['coef'] = np.abs(new_feature_coef_df['coef'])

coefs = feature_coef_df[["feature", "coef"]].groupby("feature").mean().sort_values(by="coef", ascending=False)
feat_imp = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)
coefs.head(10)
feat_imp.head(10)
oof_preds1
oof_preds2