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

import math

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
dataset.info()
dataset.shape
dataset.head()
dataset.nunique()
dataset.dropna(inplace=True)
X = dataset.iloc[:, 2:-1].values

y = dataset.iloc[:, 1].values
X[1]
y[1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state = 1,stratify =y)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
logreg = LogisticRegression()

logreg.fit(X_train,y_train)
x_test_test = test.iloc[:,2:]
y_pred_test= logreg.predict(x_test_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
y_pred_test
submission_rfc = pd.DataFrame({

        "ID_code": dataset["ID_code"],

        "target": y_pred_test

    })

submission_rfc.to_csv('logistic_regression.csv', index=False)