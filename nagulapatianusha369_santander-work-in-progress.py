# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

import lightgbm as lgbm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.describe()
train.info()
train.head()
X = train.drop(['ID_code','target'],axis = 1)

y = train['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
plt.hist(y_train)
smt = SMOTE()
X_train,y_train = smt.fit_sample(X_train,y_train)
sns.distplot(y_train)
sns.distplot(train["var_187"])
lr = LogisticRegression(verbose= 1)
lr.fit(X_train,y_train)
Lrpredictions = lr.predict(X_val)
roc_auc_score(y_val,Lrpredictions)
from sklearn.metrics import auc




fpr, tpr, threshold = roc_curve(y_val, Lrpredictions)

roc_auc = auc(fpr, tpr)







plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
plt.roc_curve(y_val,Lrpredictions)
test.head()
X_test = test.drop(['ID_code'], axis = 1)
y_test = lr.predict(X_test)
submission_LR = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_test

    })

submission_LR.to_csv('submission_rfc.csv', index=False)