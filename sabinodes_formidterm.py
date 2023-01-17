import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns



import pandas as pd

import numpy as np

from scipy.stats import skew

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor

import xgboost as xgb

from scipy import stats

from scipy.stats import norm, skew #for some statistics



import os
import pandas as pd

sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
train.head()
test.head()
train.info()
train.describe()
train.corr()
sns.countplot(x='target', data=train)
print(train.shape, test.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(7)
X, y = train.iloc[:,2:], train.iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)
knn.fit(X_train, y_train)
X_test1 = np.nan_to_num(X_test)
y_preds = knn.predict(X_test1)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix



accuracy_score(y_test, y_preds)
confusion_matrix(y_test,y_preds)
print(classification_report(y_test, y_preds))
from sklearn.svm import SVC

svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
svc_model = SVC()
svc_model.fit(X_train, y_train)
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_predit_svc = gnb.predict(X_test)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix



accuracy_score(y_test, y_predit_svc)
confusion_matrix(y_test,y_predit_svc)
print(classification_report(y_test, y_predit_svc))
y_preds_res = gnb.predict(X)
accuracy_score(y, y_preds_res)
confusion_matrix(y, y_preds_res)
print(classification_report(y, y_preds_res))
submission_nb = pd.DataFrame({

    "ID_code": test["ID_code"],

    "target": y_preds_res

})

submission_nb.to_csv('naive_baise_submission.csv', index=False)