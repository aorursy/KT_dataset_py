# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart.csv')

data.head()
data.tail()
data.target.value_counts()
X = data.iloc[:, :-1]

X.head()
y = data.iloc[:, -1]

y.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier



regg = DecisionTreeClassifier()
regg.fit(X_train, y_train)
y_pred = regg.predict(X_test)
y_pred = pd.Series(y_pred)
from sklearn.metrics import f1_score



f1_score(y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier



regg_rf = RandomForestClassifier(n_estimators = 3, random_state=42)
regg_rf.fit(X_train, y_train)
y_pred_rf = regg_rf.predict(X_test)
f1_score(y_test, y_pred_rf)