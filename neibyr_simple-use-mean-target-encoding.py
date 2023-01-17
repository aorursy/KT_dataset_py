!pip install target_encoding

# https://github.com/KirillTushin/target_encoding
from target_encoding import TargetEncoderClassifier

from target_encoding import TargetEncoder



from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score





X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



enc = TargetEncoder()

new_X_train = enc.transform_train(X=X_train, y=y_train)

new_X_test = enc.transform_test(X_test)



rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

pred = rf.predict_proba(X_test)[:,1]

print('without target encoding', roc_auc_score(y_test, pred))



rf.fit(new_X_train, y_train)

pred = rf.predict_proba(new_X_test)[:,1]

print('with target encoding', roc_auc_score(y_test, pred))



enc = TargetEncoderClassifier()

enc.fit(X_train, y_train)

pred = enc.predict_proba(X_test)[:,1]

print('target encoding classifier', roc_auc_score(y_test, pred))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from target_encoding import TargetEncoder, TargetEncoderClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import roc_auc_score



import os

print(os.listdir("../input"))
train=pd.read_csv("../input/train.csv").drop("ID_code",axis=1)

test=pd.read_csv("../input/test.csv").drop("ID_code",axis=1)



X = train.drop('target', axis=1)

y = train.target



sample_submission = pd.read_csv('../input/sample_submission.csv')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
enc = TargetEncoderClassifier(alpha=100, max_unique=25, used_features=170)

score = cross_val_score(enc, X, y, scoring='roc_auc', cv=cv)

print(score.mean(), score.std())
enc = TargetEncoderClassifier(alpha=100, max_unique=25, used_features=170)

enc.fit(X, y)

pred = enc.predict_proba(test)[:,1]
sample_submission['target'] = pred

sample_submission.to_csv('submission.csv', index=False)