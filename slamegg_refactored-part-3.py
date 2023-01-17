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
from sklearn import svm
X_train = dataset_train.iloc[:, dataset_train.columns != 'target']

y_train = dataset_train.iloc[:, 1].values

X_test = dataset_test.iloc[:, dataset_test.columns != 'ID_code'].values
X_train = X_train = X_train.iloc[:, X_train.columns != 'ID_code'].values
clf = svm.SVC()

clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)
dataset_svm = pd.concat((dataset_test.ID_code, pd.Series(clf_pred).rename('target')), axis = 1)

dataset_svm.target.value_counts()
dataset_svm.to_csv('svm_better_submission.csv', index=False)