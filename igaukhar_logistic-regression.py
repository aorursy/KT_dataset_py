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
import pandas as pd

test = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/train.csv", index_col='ID_code')
train.head()
train.isnull().sum().sum()
train.corr()
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,  cross_val_score
y = train['target']

X = train.drop(['target','ID_code'],axis=1)

X_test = test.drop('ID_code', axis=1)
from sklearn.linear_model import LogisticRegression
cr = LogisticRegression()

cr.fit(X_train, y_train)
y_pred = cr.predict(X_train)
y_pred.shape
from sklearn.metrics import *
confusion_mat = confusion_matrix(y_true=y_test, y_pred = y_pred)

confusion_mat
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
pd.concat([test.ID_code, pd.Series(y_pred).rename('target')], axis = 1).to_csv('submission_log.csv', index =False)
from sklearn.svm import SVC
svc = SVC()
#svc.fit(X_train,y_train)
#y_pred=svc.predict(X_test)
#confusion_mat = confusion_matrix(y_true=y_test, y_pred = y_pred)

#confusion_mat
#accuracy_score(y_test, y_pred)
#print(classification_report(y_test, y_pred))
#pd.concat([test.ID_code, pd.Series(y_pred).rename('target')], axis = 1).to_csv('submission_svm.csv', index =False)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(X_train,y_train)
prd = nb.predict(X_test)
cm = confusion_matrix(y_true=y_test,y_pred=prd)

cm
print(accuracy_score(y_test,prd))

print(precision_score(y_test,prd,average='macro'))

print(recall_score(y_test,prd,average='macro'))
print(classification_report(y_test, prd))
pd.concat([test.ID_code, pd.Series(prd).rename('target')], axis = 1).to_csv('submission_naive.csv', index =False)