# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_iris



# save "bunch" object containing iris dataset and its attributes

iris = load_iris()



# store feature matrix in "X"

X = iris.data



# store response vector in "y"

y = iris.target
# print the shapes of X and y

print(X.shape)

print(y.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

print(knn)

knn.fit(X, y)



X_new = [[3, 5, 4, 4], [2, 4, 3, 2]]

knn.predict(X_new)