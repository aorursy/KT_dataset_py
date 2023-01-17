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

train_ds.head()
test_ds.head()
train_ds.shape
x = train_ds.iloc[:, 2:].values

y = train_ds.iloc[:, 1].values
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10, stratify = y)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from imblearn.over_sampling import SMOTE

logreg = LogisticRegression()

logreg.fit(x_train, y_train)
test = test_ds.iloc[:, 1:].values

y_pred = logreg.predict(test)





# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

result = pd.DataFrame({"target" : np.array(y_pred).T})

# train_ds['target'].value_counts()

result['target'].value_counts()
train_ds['target'].value_counts()
# print(classification_report(y_test,y_pred))

y_pred.shape
sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

sub['target'] = y_pred

sub.to_csv('predictionn.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(11)



knn.fit(x_train, y_train)



y_preds = knn.predict(x_test)