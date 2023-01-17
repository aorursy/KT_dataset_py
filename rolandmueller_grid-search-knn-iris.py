# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
%matplotlib inline

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
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_test, y_test)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
params = {
    'n_neighbors' : [1, 2, 3, 4, 5, 6, 7, 8, 10, 25],
    'weights': ['uniform', 'distance']
}

grid_clf = GridSearchCV(estimator = clf,
                        param_grid = params,
                        scoring = 'accuracy', 
                        cv = 10, 
                        verbose = 1,
                        n_jobs = -1)
grid_clf.fit(X_train, y_train)
grid_clf.best_estimator_
# Identify optimal hyperparameter values
best_n_neighbors = grid_clf.best_params_['n_neighbors']  
best_weights      = grid_clf.best_params_['weights']


print(f"Beste Cross-Validation Klassifikationsgenauigkeit: {grid_clf.best_score_:.3f}")
print(f"Bester n_neighbors Wert: {best_n_neighbors}")
print(f"Bester weights Wert: {best_weights}")
grid_clf.score(X_test, y_test)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3, weights='uniform')
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_predict = clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict,target_names=iris.target_names))
from yellowbrick.classifier import ConfusionMatrix
iris_cm = ConfusionMatrix(
    clf, classes=iris.target_names,
    label_encoder={0: 'setosa', 1: 'versicolor', 2: 'virginica'}
)
iris_cm.score(X_test, y_test)
iris_cm.show()

