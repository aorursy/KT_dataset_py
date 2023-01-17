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
x = train_ds.iloc[:, 2:].values

y = train_ds.iloc[:, 1].values
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.00001, random_state=0)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

# clf.partial_fit(x_train, y_train, np.unique(y_train))

kfold = KFold(n_splits=150, random_state=0, shuffle=False)

results = cross_val_score(clf, x, y, cv=kfold)
for train_index, test_index in kfold.split(x):

 x_train, x_test = x[train_index], x[test_index]

 y_train, y_test = y[train_index], y[test_index]
clf.partial_fit(x_train, y_train, np.unique(y_train))
test = test_ds.iloc[:, 1:].values

y_preds = clf.predict(test)

result = pd.DataFrame({"target" : np.array(y_preds).T})

# train_ds['target'].value_counts()

result['target'].value_counts()
sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

sub['target'] = y_preds

sub.to_csv('gauss2.csv', index=False)