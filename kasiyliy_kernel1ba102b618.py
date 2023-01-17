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
dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

dataseTest = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

dataset.sample(5)
X = dataset.iloc[:, 2:].values

Y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state = 10,stratify =Y)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
test = dataseTest.iloc[:, 1:].values
y_pred = clf.predict(test)
sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')

sub['target'] = y_pred

sub.to_csv('DECISION_TREE.csv', index = False)