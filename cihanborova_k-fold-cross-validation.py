

import numpy as np

import pandas as pd 

from sklearn.datasets import load_iris



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



iris=load_iris()
x = iris.data

y = iris.target
x = (x - np.min(x)) / (np.max(x) - np.min(x))
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state = 1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=knn , cv=9,X=x_train,y=y_train)
print("Avarage accuracy: ",np.mean(accuracies))

print("Avarage std: ",np.std(accuracies))

knn.fit(x_train,y_train)

print ("Accuracy: " , knn.score(x_test,y_test))
from sklearn.model_selection import GridSearchCV
k=KNeighborsClassifier()
grid = {"n_neighbors":np.arange(1,50) }

knn_cv = GridSearchCV(knn,grid,cv=10)
knn_cv.fit(x,y)

print("Tuned hyperparamater K= ",knn_cv.best_params_)

print("Paramatreye göre en iyi score = ",knn_cv.best_score_)