# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test_data = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
print(train_data.shape)

print(test_data.shape)
print(train_data.columns)

print(test_data.columns)
train_data.head()
test_data.head()
#Since our target column is in binary format, so here we can apply various Classification model to get our best accuracy
target_ = train_data['target']
from sklearn.preprocessing import LabelEncoder
len_uniques = []

for c in train_data.columns.drop(['target','id']):

    le = LabelEncoder()

    le.fit(pd.concat([train_data[c], test_data[c]])) 

    train_data[c] = le.transform(train_data[c])

    test_data[c] = le.transform(test_data[c])

    len_uniques.append(len(le.classes_))

print("train data.shape: {}  test data.shape: {}".format(train_data.shape, test_data.shape))
X = train_data.drop(['target', 'id'], axis=1)

y = train_data['target']
print("X.shape: {}  y.shape: {}".format(X.shape, y.shape))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.20, random_state=42) 

logreg = LogisticRegression() 

logreg.fit(X_train, y_train)

log_pre=logreg.predict(X_test)

print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))

print('Accuracy : ',accuracy_score(y_test,log_pre))

rfcl = RandomForestClassifier(n_estimators=70, n_jobs=-1, min_samples_leaf=5)

rfcl.fit(X_train, y_train)

rfcl_pre=rfcl.predict(X_test)

print("Test score: {:.2f}".format(rfcl.score(X_test, y_test)))

print('Accuracy : ',accuracy_score(y_test,rfcl_pre))
gbcl = GradientBoostingClassifier()

gbcl.fit(X_train, y_train)

gbcl_pre=gbcl.predict(X_test)

print("Test score: {:.2f}".format(gbcl.score(X_test, y_test)))

print('Accuracy : ',accuracy_score(y_test,gbcl_pre))
from sklearn.model_selection import StratifiedKFold 

from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_score(gbcl, X_train, y_train, cv=skfolds, scoring="accuracy") 
test_d = test_data.set_index('id')
test_d.head()
sample_submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv', index_col='id')
sample_submission['target'] = gbcl.predict_proba(test_d)[:, 1]
sample_submission.head(10)
"""svc_clf = SVC(kernel='sigmoid', gamma=2)

svc_clf.fit(X_train, y_train)

svc_pre=svc_clf.predict(X_test)

print("Test score: {:.2f}".format(svc_clf.score(X_test, y_test)))

print('Accuracy : ',accuracy_score(y_test,svc_pre_pre))"""
sample_submission.to_csv('sample_submission_sunnyc.csv')