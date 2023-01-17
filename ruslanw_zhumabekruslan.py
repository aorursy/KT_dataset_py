# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

test_ds = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
train_ds.head()
train_ds.columns
train_ds.describe
train_ds.info()
train_ds.isna().sum()
train_ds.nunique()
X = train_ds.iloc[:, 2:].values

y = train_ds.iloc[:, 1].values
X[0]
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
logreg = LogisticRegression(max_iter=10000)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)



print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
test = test_ds.iloc[:, 1:].values

y_pred = logreg.predict(test)



print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

submission['target'] = y_pred

submission.to_csv('logreg.csv', index=False)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, stratify = y)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import classification_report, confusion_matrix



target_names = ['0', '1']



print(classification_report(y_test, y_pred, target_names=target_names))
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)
y_pred = gnb.predict(test)
sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

sub['target'] = y_pred

sub.to_csv('gaussian.csv', index=False)
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split
bnb = BernoulliNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, stratify = y)
bnb.fit(X_train, y_train)

bnb.score(X_test, y_test)

y_pred = bnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=target_names))
y_pred = bnb.predict(test)
sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

sub['target'] = y_pred

sub.to_csv('bernoulli.csv', index=False)
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics 

from sklearn.tree import export_graphviz 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, stratify = y)
clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
y_pred = clf.predict(test)
sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

sub['target'] = y_pred

sub.to_csv('decisionTree.csv', index=False)
import xgboost as xgb

import pandas as pd
xg_cl = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=1,

              learning_rate=0.1, max_delta_step=0, max_depth=8,

              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10, stratify = y)
xg_cl.fit(X_train, y_train)
y_pred = xg_cl.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))
y_pred = xg_cl.predict(test)
sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

sub['target'] = y_pred

sub.to_csv('xgboost.csv', index=False)
dataset_dmatrix = xgb.DMatrix(data = X,label = y)

dataset_dmatrix
params = {"objective":"reg:logistic", "max_depth":3}

params
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)
print(cv_results)

print(1-cv_results["test-rmse-mean"].tail(1))
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)
print(cv_results["test-auc-mean"].tail(1))