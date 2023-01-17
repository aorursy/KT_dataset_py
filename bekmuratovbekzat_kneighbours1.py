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

from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection



X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.65,test_size=0.35, random_state=101)

print ("X_train: ", X_train)

print ("y_train: ", y_train)

print("X_test: ", X_test)

print ("y_test: ", y_test)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 1284,test_size = 0.99358, random_state = 80, stratify = y)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()



knn.fit(x_train, y_train)



y_preds = knn.predict(x_test)
result = pd.DataFrame({"target" : np.array(y_preds).T})

# train_ds['target'].value_counts()

result['target'].value_counts()