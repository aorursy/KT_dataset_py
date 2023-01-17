# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pickle

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as plt

from sklearn.metrics import roc_auc_score, roc_curve



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
print('Train:', train.shape)

print('Test:', test.shape)
train.info()

train.isna().sum()

train.describe()

train.corr
sns.countplot(train['target'])
Target = train['target']

train_inp = train.drop(columns = ['target', 'ID_code'])

test_inp = test.drop(columns = ['ID_code'])
X_train, X_test, y_train,  y_test = train_test_split(train_inp, Target,test_size=0.5, random_state=40)
logist = LogisticRegression(class_weight='balanced',max_iter=25)

logist.fit(X_train, y_train)
logist_pred = logist.predict_proba(X_test)[:,1]
logist_pred
def performance(Y_test, logist_pred):

    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]

    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)

    print('AUC:')

    print(auc(fpr, tpr))
performance(y_test, logist_pred)
logist_pred_test = logist.predict_proba(test_inp)[:,1]

submit = test[['ID_code']]

submit['target'] = logist_pred_test

submit.head()
submit.to_csv('log_reg_kyzyrov.csv', index = False)