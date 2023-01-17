# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Iris.csv")
data.head()
data["Species"].value_counts()
data.describe()
X = data.drop(["Id", "Species"], axis=1)
y = data["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    test_size=0.2)
rf_clf = RandomForestClassifier(random_state=42)
mlp_clf = MLPClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
mlp_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
rf_cm = confusion_matrix(y_test, y_pred)
print(rf_clf.score(X_test, y_test))
print(rf_cm)
y_pred = mlp_clf.predict(X_test)
mlp_cm = confusion_matrix(y_test, y_pred)
print(mlp_clf.score(X_test, y_test))
print(mlp_cm)
