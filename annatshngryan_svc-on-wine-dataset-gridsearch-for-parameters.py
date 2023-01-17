# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_wine
data = load_wine()
print(data['DESCR'])
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['Target'] = data['target']
df.head()
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
svc = LinearSVC(random_state=42)
svc.fit(X_train, y_train)
test_predictions = svc.predict(X_test)
train_predictions = svc.predict(X_train)
print("Train:")
print(classification_report(y_train, train_predictions))
print("Test:")
print(classification_report(y_test, test_predictions))
svc2 = SVC(random_state=42)
svc2.fit(X_train, y_train)
test_predictions = svc2.predict(X_test)
train_predictions = svc2.predict(X_train)
print("Train:")
print(classification_report(y_train, train_predictions))
print("Test:")
print(classification_report(y_test, test_predictions))
gamma = 1/np.power(10, np.arange( 10))
C = np.power(10, np.arange( 10))
params = {'C':C,
         'gamma': gamma,}
grid = GridSearchCV(SVC(), params, scoring="f1_macro",
                    cv=3, verbose=1)
grid.fit(X, y)
grid.best_params_
test_predictions = grid.predict(X_test)
train_predictions = grid.predict(X_train)
print("Train:")
print(classification_report(y_train, train_predictions))
print("Test:")
print(classification_report(y_test, test_predictions))
