# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import category_encoders as ce

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
train.sort_index(inplace=True)

train_y = train['target']; test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True); test.drop('id', axis=1, inplace=True)
folds = 20

smoothing=100

cat_feat_to_encode = train.columns[:].tolist()



oof = np.zeros(train.shape)

test_oof = np.zeros(test.shape)

from sklearn.model_selection import StratifiedKFold

for tr_idx, oof_idx in StratifiedKFold(n_splits=folds, random_state= 1032, shuffle=True).split(train, train_y):

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])

    oof[oof_idx,:] = ce_target_encoder.transform(train.iloc[oof_idx, :]).values

    test_oof[:,:] += ce_target_encoder.transform(test).values

test_oof /= folds

train = pd.DataFrame(data=oof,columns=cat_feat_to_encode,index=train.index.values)

test = pd.DataFrame(data=test_oof,columns=cat_feat_to_encode,index=test.index.values)   

glm =LogisticRegression( random_state=2, solver='lbfgs', max_iter=20600, fit_intercept=True, penalty='l2', verbose=0)

glm.fit(train, train_y)
roc_auc_score(train_y,glm.predict_proba(train)[:,1])
pd.DataFrame({'id': test_id, 'target': glm.predict_proba(test)[:,1]}).to_csv('submission.csv', index=False)