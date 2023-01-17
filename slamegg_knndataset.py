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

import numpy as np

import matplotlib.pyplot as plt
dataset_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

dataset_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

dataset_train.head()
dataset_test.head()
dataset_train.info()
X_train = dataset_train.iloc[:, dataset_train.columns != 'target'].values

y_train = dataset_train.iloc[:, 1].values

X_test = dataset_test.values
dataset_train.describe()


dataset_train.target.value_counts() 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



X_train[:,0] = le.fit_transform(X_train[:,0])

X_test[:,0] = le.fit_transform(X_test[:,0])

knn = KNeighborsClassifier(11)
knn.fit(X_train, y_train)
y_preds = knn.predict(X_test)
y_preds
pd.concat([dataset_test.ID_code, pd.Series(y_preds).rename('target')], axis = 1).to_csv('knn_submission.csv')
result_dataset = pd.concat([dataset_test.ID_code, pd.Series(y_preds).rename('target')], axis = 1)
result_dataset.target.value_counts()
# Logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg_cls = LogisticRegression()
log_reg_cls.fit(X_train, y_train)
y_preds_log_reg = log_reg_cls.predict(X_test)
y_preds_log_reg.shape
y_preds_log_reg
pd.concat([dataset_test.ID_code, pd.Series(y_preds_log_reg).rename('target')], axis = 1).to_csv('log_reg_submission.csv')
log_reg_dataset = pd.concat([dataset_test.ID_code, pd.Series(y_preds_log_reg).rename('target')], axis = 1)

log_reg_dataset.target.value_counts()
from sklearn import svm
clf = svm.SVC()

clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)
clf_pred
svm_dataset = pd.concat([dataset_test.ID_code, pd.Series(clf_pred).rename('target')], axis = 1)

svm_dataset.target.value_counts()
svm_dataset.to_csv('svm_submission.csv')