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
insurance2 = pd.read_csv('../input/sample-insurance-claim-prediction-dataset/insurance2.csv', header=0)
insurance3 = pd.read_csv('../input/sample-insurance-claim-prediction-dataset/insurance3r2.csv', header=0)
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = insurance3.drop(columns=['insuranceclaim'])
Y = insurance3['insuranceclaim']
acc = []
for _ in range(100):
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    acc.append(accuracy)
    
print("Accuracy: %.2f%%" % (np.mean(accuracy) * 100.0))
X = insurance2.drop(columns=['insuranceclaim'])
Y = insurance2['insuranceclaim']
acc = []
for _ in range(100):
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    acc.append(accuracy)
    
print("Accuracy: %.2f%%" % (np.mean(accuracy) * 100.0))