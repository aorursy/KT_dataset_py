# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

ds = load_iris()

#features 
ds.data

#leabel
ds.target_names

#features data
X  = np.array(ds.data)

#leabel data
y = np.array(ds.target)

X_train, X_test, y_train,  y_test = train_test_split(X, y)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

kpredict = knn.predict([[ 5.6 , 2.9,  3.6 , 1.3]])

kpredict

kscore = knn.score(X_train, y_train)

print (kpredict)
print (kscore)

