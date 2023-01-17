# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
training_set  = pd.read_csv('../input/train.csv')
training_set.columns
plt.hist(training_set['target'].values)
X = training_set[training_set.columns.values[2:]]

X.head()
from sklearn.datasets import load_boston

from sklearn.ensemble import RandomForestRegressor

import numpy as np

Y = training_set["target"]



X = training_set[training_set.columns.values[2:]]

names = X.columns.values

rf = RandomForestRegressor()

rf.fit(X, Y)

print("Features sorted by their score:")

print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 

             reverse=True))
test = pd.read_csv('../input/test.csv')

test.head()
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(7, 2), random_state=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.20, random_state=1)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


scores = cross_val_score(clf,X,Y, cv=5)

clf.fit(X, Y) 
test_X = test[test.columns.values[1:]]
predictions = clf.predict(test_X)
predictions
submission = pd.DataFrame({"ID_code":test["ID_code"].values})

submission["target"] = predictions

submission.to_csv("submission.csv", index=False)