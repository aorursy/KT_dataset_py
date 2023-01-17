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
from sklearn.naive_bayes import GaussianNB

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train.head()
test.head()
print('Train:', train.shape)

print('Test:', test.shape)
sns.countplot(train['target'])
train['target'].value_counts()
def performance(Y_test, logist_pred):

    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]

    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)

    print('AUC:')

    print(auc(fpr, tpr))
Target = train['target']

train_inp = train.drop(columns = ['target', 'ID_code'])

test_inp = test.drop(columns = ['ID_code'])
X_train, X_test, y_train,  y_test = train_test_split(train_inp, Target,test_size=0.5, random_state=1999)
print('Train:',X_train.shape)

print('Test:',X_test.shape)

print('Train:',y_train.shape)

print('Test:',y_test.shape)
nb = GaussianNB()
nb.fit(X_train,y_train)
nb_pred = nb.predict_proba(X_test)[:, 1]

performance(y_test, nb_pred)
nb_pred
nb_pred_test = nb.predict_proba(test_inp)[:,1]

submit = test[['ID_code']]

submit['target'] = nb_pred_test

submit.head()
submit.to_csv('NB_baseline.csv', index = False)