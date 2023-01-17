
from __future__ import division, print_function
import numpy as np
from sklearn import datasets, svm 
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
kernels = ['linear','poly','rbf']
accuracies = []
for kernel in kernels:
    model = svm.SVC(kernel=kernel)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    accuracies.append(acc)
    print("{} accuracy: {}".format(kernel, acc))