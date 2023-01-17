import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.metrics import precision_score, recall_score, accuracy_score,classification_report

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import resample

from sklearn.utils.testing import ignore_warnings

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.linear_model import LogisticRegression 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/Santander/"

else:

    PATH="../input/"

os.listdir(PATH)
%%time

train = pd.read_csv(PATH+"train.csv")

test= pd.read_csv(PATH+"test.csv")
train.head()
test.head()
sns.countplot(train.target)
sns.distplot(train[train.target == False]['var_37'], hist=False)

sns.distplot(train[train.target == True]['var_37'], hist=False)
train_float = train.select_dtypes(include=['float'])
converted_train = train_float.apply(pd.to_numeric,downcast='float')
converted_train
def mem_usage(pandas_obj):

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else: # we assume if not a df it's a series

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes

    return "{:03.2f} MB".format(usage_mb)
print(mem_usage(train_float))

print(mem_usage(converted_train))
converted_train['ID_code'] = train['ID_code']
converted_train['target'] = train['target']
converted_train
Y = converted_train.iloc[:,-1]

X = converted_train.iloc[:,0:200]
train.iloc[:,1]

X_test2 = test.iloc[:,1:]
print(Y.shape)

print(X.shape)

print(X_test2.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
xgb_cl = xgb.XGBClassifier()
xgb_cl.fit(X_train, y_train)
y_pred_xgb = xgb_cl.predict(X_test)
print("Precision = {}".format(precision_score(y_test, y_pred_xgb, average='macro')))

print("Recall = {}".format(recall_score(y_test, y_pred_xgb, average='macro')))

print("Accuracy = {}".format(accuracy_score(y_test, y_pred_xgb)))
print(classification_report(y_test, y_pred_xgb))
y_pred_xgb_test = xgb_cl.predict(X_test2)
submission_xgb = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_xgb_test

    })

submission_xgb.to_csv('submission_xgb.csv', index=False)
xgb.plot_importance(xgb_cl)

plt.rcParams['figure.figsize'] = [1,51]

plt.show()
param_grid = {'max_depth': [5,6,7,8], 'gamma': [1, 2, 4], 'learning_rate': [1, 0.1, 0.01, 0.001]}
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
target_names = ['False', 'True']
print(classification_report(y_test, y_pred, target_names=target_names))
y_pred_gnb_test = gnb.predict(X_test2)
submission_gnb = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_gnb_test

    })

submission_gnb.to_csv('submission_gnb.csv', index=False)
from sklearn.naive_bayes import BernoulliNB
bnb  = BernoulliNB(binarize=0.0)
bnb.fit(X_train, y_train)
y_pred_bnb =bnb.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred_bnb))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred_bnb)

print(confusion_matrix)
print(classification_report(y_test, y_pred_bnb, target_names=target_names))
y_pred_bnb_test = bnb.predict(X_test2)
submission_bnb = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_bnb_test

    })

submission_bnb.to_csv('submission_bnb.csv', index=False)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
y_pred_logreg_test = logreg.predict(X_test2)
submission_logreg = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_logreg_test

    })

submission_logreg.to_csv('submission_logreg.csv', index=False)