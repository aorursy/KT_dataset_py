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
train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
train
test
train.describe()
train.info()
train.loc[train['target'] == 1].describe()
corr_values = train.corr()

corr_values
target_correlations = corr_values['target'].abs().sort_values(ascending=False)

target_correlations
from sklearn.metrics import roc_auc_score
y = train.target

X = train.drop(columns = ['ID_code', 'target'])

test.drop(columns='ID_code', inplace=True)
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2, random_state = 0)
# LogReg



from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()

logReg.fit(X_train, Y_train)
y_pred = logReg.predict(X_test)

roc_auc_score(Y_test, y_pred)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test, y_pred))
confusion_matrix(Y_test, y_pred)
from sklearn.naive_bayes import GaussianNB
nbg = GaussianNB()

nbg.fit(X_train,Y_train)

roc_auc_score(Y_test, nbg.predict(X_test))
from sklearn.metrics import accuracy_score

accuracy_score(Y_test,nbg.predict(X_test))
print(classification_report(Y_test, nbg.predict(X_test)))
confusion_matrix(Y_test, nbg.predict(X_test))
# DecisionTree

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

accuracy_score(Y_test, dt.predict(X_test))

roc_auc_score(Y_test, dt.predict(X_test))
confusion_matrix(Y_test, dt.predict(X_test))
print(classification_report(Y_test, dt.predict(X_test)))
# RandomForest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, Y_train)

accuracy_score(Y_test, rf.predict(X_test))

roc_auc_score(Y_test, rf.predict(X_test))
confusion_matrix(Y_test, rf.predict(X_test))
print(classification_report(Y_test, rf.predict(X_test)))
# XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, Y_train)

accuracy_score(Y_test, xgb.predict(X_test))

roc_auc_score(Y_test, xgb.predict(X_test))
confusion_matrix(Y_test, xgb.predict(X_test))
print(classification_report(Y_test, xgb.predict(X_test)))