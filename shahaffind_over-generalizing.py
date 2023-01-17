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

train.shape
X = train.drop(['id', 'target'], axis=1).values

y = train['target']
from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import RFE

import lightgbm as lgb

from sklearn.svm import SVC, SVR

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier



n_fold = 5



kf = KFold(n_splits=n_fold, shuffle=True, random_state=2)

scores = []



for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    clf = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=.1, solver='liblinear')

    model = BaggingClassifier(clf, n_estimators=300, max_samples=.8, max_features=.8, random_state=0)

    model.fit(X_train, y_train)

    

    y_pred = model.predict_proba(X_test)[:, 1]

    scores.append(roc_auc_score(y_test, y_pred))



print(f'CV mean score: {np.mean(scores):.4f}')
X_test = test.drop(['id'], axis=1).values



log_reg = linear_model.LogisticRegression(class_weight='balanced', penalty='l1', C=.1, solver='liblinear')

model = BaggingClassifier(log_reg, n_estimators=300, max_samples=.8, max_features=.8, random_state=0)

model.fit(X, y)



y_pred = model.predict_proba(X_test)[:, 1]
sub = pd.read_csv("../input/sample_submission.csv")

sub['target'] = y_pred
sub.to_csv('submission.csv',index=False)