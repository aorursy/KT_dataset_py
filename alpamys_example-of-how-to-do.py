

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
X = train_df.drop(['id_code','target'], axis=1)

y = train_df['target']
X_dummy = pd.get_dummies(X)
logreg = LogisticRegression()

logreg.fit(X_dummy,y)
y_pred_proba = logreg.predict_proba(X_dummy)[:,1]    # we get only prediction for positive class
roc_auc_score(y, y_pred_proba)
test_df_dummy = pd.get_dummies(test_df.drop('id_code', axis=1))
y_pred_test = logreg.predict_proba(test_df_dummy)[:,1]
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = y_pred_test
submission.to_csv('funny_submission.csv', index=False)