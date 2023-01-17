# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_iris
iris = load_iris()
iris.data
X = iris.data
type(X)
print("Shape of X:", X.shape)
iris.feature_names
iris.target
y = iris.target
type(y)
print("Shape of y:", y.shape)
iris.target_names
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)
print("Shape of X_train:", X_train.shape)

print("Shape of X_test:", X_test.shape)

print("Shape of y_train:", y_train.shape)

print("Shape of y_test:", y_test.shape)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.metrics import ConfusionMatrixDisplay

cmd = ConfusionMatrixDisplay(cm, iris.target_names)

cmd.plot()
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)

k_range = range(1,60)

accuracy_list = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)

    print(f"k={k} ergibt eine Klassifikationsgenauigkeit von {accuracy}")

    accuracy_list.append(accuracy)
%matplotlib inline

import matplotlib.pyplot as plt



plt.plot(k_range, accuracy_list)

plt.xlabel("k nächste Nachbarn")

plt.ylabel("Klassifikationsgenauigkeit")

plt.show()