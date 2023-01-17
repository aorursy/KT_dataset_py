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
from sklearn.datasets import load_iris
iris=load_iris()

x=iris.data
y=iris.target
# normalization
x=(x-np.min(x))/(np.max(x)-np.min(x))

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)
# knn model
from sklearn.neighbors  import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=3)
#K FOLD CV K=10
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=knn,X=x_train,y=y_train,cv=10)
accuracies
print("average accuracy :",np.mean(accuracies))
print("average std :",np.std(accuracies))

# test your model
knn.fit(x_train,y_train)
print("test accuracy :",knn.score(x_test,y_test))
# Grid search cross validation
from sklearn.model_selection import GridSearchCV

grid ={"n_neighbors":np.arange(1,50)}
knn= KNeighborsClassifier()
knn_cv=GridSearchCV(knn,grid,cv=10) #GridSearchCV
knn_cv.fit(x,y)
# print hyperparameter KNN algoritmasondaki K değeri
print("tuned hyperparameter K:",knn_cv.best_params_)
print("tuned parametreye göre en iyi accuracy (best score):",knn_cv.best_score_)