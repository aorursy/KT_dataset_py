# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from IPython.display import IFrame

from sklearn.datasets import load_iris
iris = load_iris()

type(iris)
print(iris.data)
print(iris.target)
print(iris.target_names)
print(type(iris.data))

print(type(iris.target))
print(iris.data.shape)
X = iris.data

y = iris.target
print(X.shape)

print(y.shape)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

print(knn)
knn.fit(X,y)
knn.predict([[3,5,4,2]])
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X,y)

logreg.predict(X_new)
logreg.predict(X)
knn.predict(X)
y_pred = logreg.predict(X)

len(y_pred)
from sklearn import metrics

print(metrics.accuracy_score(y,y_pred))
y_pred_knn = knn.predict(X)
print(metrics.accuracy_score(y,y_pred))
from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 4)
print(X_train.shape)

print(X_test.shape)
print(y_train.shape)

print(y_test.shape)
logreg = LogisticRegression()

logreg
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred))
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier(n_neighbors=25)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
k_range = list(range(1,26))

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors= k)

    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test,y_pred))
scores
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(k_range,scores)

plt.xlabel('value of K for KNN')

plt.ylabel('Testing accuracy')
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

print(scores)

print(scores.mean())
# Search for optimal value of K for KNN

k_scores = []

k_range = list(range(1,31))
for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')

    k_scores.append(scores.mean())

print(k_scores)
plt.plot(k_range,k_scores)

plt.xlabel('values of K for KNN')

plt.ylabel('Cross validated Accuracy')
# Apply 10 fold cross validation with best KNN model

knn= KNeighborsClassifier(n_neighbors=20)

print(cross_val_score(knn,X,y,cv=10,scoring='accuracy').mean())
# 10 fold using logistic regression

print(cross_val_score(LogisticRegression(),X,y,cv=10,scoring='accuracy').mean())