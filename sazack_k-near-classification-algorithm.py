import sklearn
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state = 45)
neigh = KNeighborsClassifier(n_neighbors = 5)
neigh.fit(X_train,y_train)

print(neigh.predict(X_test))
prediction = neigh.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(prediction,y_test)
print(accuracy)
wines = datasets.load_wine()
print(wines.data)

X = wines.data
y = wines.target

features_train, features_test, labels_train, labels_test = train_test_split(X,y, test_size = 0.30, random_state = 42)
neigh = KNeighborsClassifier(n_neighbors = 7)
neigh.fit(features_train, labels_train)
prediction = neigh.predict(features_test)

accuracy = sklearn.metrics.accuracy_score(prediction,labels_test)
print(accuracy)