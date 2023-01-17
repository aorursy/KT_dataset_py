import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, roc_auc_score

from sklearn import metrics

from sklearn.svm import SVC

import xgboost

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test.head()
train.head()
print('Test size = ',test.shape)

print('Train size = ',train.shape)
train.corr()
train.isnull().any().any()
train.target.value_counts()
sns.countplot(train['target'])
train_1 = train.loc[train.target ==1]

train_0 = train.loc[train.target ==0]

print("Number of target value as 1 %d" %len(train_1))

print("Number of target value as 0 {}".format(len(train_0)))
sns.distplot(train['var_0']);
x = train.drop(columns=['ID_code','target'])

y = train['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
log_reg = LogisticRegression(solver = 'lbfgs', random_state = 0)
log_reg.fit(x_train, y_train)

y_pred_logistic = log_reg.predict(x_test)
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred_logistic))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_logistic))
print("Classification report:\n", classification_report(y_test, y_pred_logistic))
roc_auc_score(y_test, y_pred_logistic)
gnb = GaussianNB()

gnb.fit(x_train, y_train)

y_pred_gnb = gnb.predict(x_test)
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred_gnb))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_gnb))
print("Classification report:\n", classification_report(y_test, y_pred_gnb))
roc_auc_score(y_test, y_pred_gnb)
dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred_dt))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification report:\n", classification_report(y_test, y_pred_dt))
roc_auc_score(y_test, y_pred_dt)
rf = RandomForestClassifier()

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred_rf))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification report:\n", classification_report(y_test, y_pred_rf))
roc_auc_score(y_test, y_pred_rf)
xgb = xgboost.XGBClassifier()

xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)
print("Accuracy score:", metrics.accuracy_score(y_test, y_pred_xgb))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification report:\n", classification_report(y_test, y_pred_xgb))
roc_auc_score(y_test, y_pred_xgb)