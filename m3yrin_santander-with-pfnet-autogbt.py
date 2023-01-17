!pip install git+https://github.com/pfnet-research/autogbt-alt.git
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from autogbt import AutoGBTClassifier
train = pd.read_csv("../input/train.csv")



y = train['target'].values

X = train.drop(['ID_code','target'], axis=1).values



train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.1)
model = AutoGBTClassifier(n_jobs=4)

model.fit(train_X, train_y)
print('valid AUC: %.3f' % (roc_auc_score(valid_y, model.predict_proba(valid_X))))

print('CV AUC: %.3f' % (model.best_score))
test = pd.read_csv('../input/test.csv')

sub = pd.DataFrame({"ID_code": test.ID_code.values})



test_X = test.drop(['ID_code'], axis=1).values



sub["target"] = model.predict_proba(test_X)

sub.to_csv("submission_autogbt.csv", index=False)