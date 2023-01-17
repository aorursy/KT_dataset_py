# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/iris/Iris.csv")
data.info()
data.head()
data.drop("Id",axis = 1 , inplace = True)
data.columns
data.Species.unique()
data.Species=[0 if i == "Iris-setosa" else 1 if  i ==  "Iris-versicolor" else 2 for i in data.Species]
data.Species.unique()
y = data.Species.values

X = data.drop("Species",axis = 1).values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
k_range = list(range(1, 26))

scores = []

for i in k_range:

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)

plt.show()
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=False).split(range(25))
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))

for iteration, data in enumerate(kf, start=1):

    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

print(scores)
print(scores.mean())
k_range = list(range(1, 31))

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

    k_scores.append(scores.mean())

print(k_scores)

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-Validated Accuracy')

plt.show()
knn = KNeighborsClassifier(n_neighbors=20)

print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())
logreg = LogisticRegression()

print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))

print(k_range)
param_grid = dict(n_neighbors=k_range)

print(param_grid)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)

grid.fit(X, y)
grid_mean_scores = grid.cv_results_['mean_test_score']

print(grid_mean_scores)
plt.plot(k_range, grid_mean_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-Validated Accuracy')
print(grid.best_score_)

print(grid.best_params_)

print(grid.best_estimator_)
k_range = list(range(1, 31))

weight_options = ['uniform', 'distance']

param_grid = dict(n_neighbors=k_range, weights=weight_options)

print(param_grid)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)

grid.fit(X, y)
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
print(grid.best_score_)

print(grid.best_params_)