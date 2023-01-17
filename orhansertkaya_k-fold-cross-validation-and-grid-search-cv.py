# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target
x = (x-np.min(x))/(np.max(x)-np.min(x))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)# k = n_neighbors
#K Fold CV K=10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn, X=x_train, y=y_train, cv=10)
accuracies
print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))
knn.fit(x_train,y_train)
print("test accuracy: ",knn.score(x_test,y_test))
#grid search cross validation for knn
from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=10)#GridSearchCV
knn_cv.fit(x_train,y_train)

#%% print hyperparameter KNN algoritmasindaki K degeri
print("tuned hyperparameter K: ",knn_cv.best_params_)
print("tuned parametreye göre en iyi accuracy (best score): ",knn_cv.best_score_)
x = x[:100,:]
y = y[:100]
from sklearn.linear_model import LogisticRegression
#"C":Logistic Regression regularization parameter
#if it is small it will be underfit.if it is big it will be overfit.
#we need to choose good value for C.
#loss functions
grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} # l1 = lasso ve l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=10)
logreg_cv.fit(x_train,y_train)

print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)
logreg2 = LogisticRegression(C=100,penalty="l1")
logreg2.fit(x_train,y_train)
print("score: ",logreg2.score(x_test,y_test))