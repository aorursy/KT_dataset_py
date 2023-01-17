# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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

import numpy as np 
iris = load_iris()

x = iris.data

y = iris.target
# Normalization

x = (x - np.min(x))/(np.max(x) - np.min(x))
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
# %% knn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # With n_neighbors = 3, we show how many times we will divide the data.
# K Fold CV

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = knn, X = x_train, y = y_train, cv = 10)

# estimator = knn means use knn algorithm when doing cross validation.

# We divided our train data with cv = 10 to 10. 9 will train and 1 validation.



print("average accurecy: ", np.mean(accuracies)) 

print("average std: ", np.std(accuracies)) # we look at the data spread to see if it is consistent or not.
knn.fit(x_train, y_train)

print("test accuracy: ",knn.score(x_test, y_test))

# we test after fit.
# Grid Search Cross Validation

# Thanks to grid search, we train according to each knn value and we do cross validation.

from sklearn.model_selection import GridSearchCV # import GridSearchCV with sklearn.

grid = {"n_neighbors":np.arange(1,50)} # We want all the neighbors in order from 1 to 50.

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10) # This is the GridSearchCV method.

knn_cv.fit(x,y)
# Print hyperparameter K value in KNN algorithm

print("tuned hyperparameter K: ", knn_cv.best_params_) # For the best neighbor value.

print("best accuracy according to the tuned parameter (best score): ",knn_cv.best_score_) # For the best score
# Grid Search CV with Logistic Regression example



x = x[:100,:]

y = y[:100]

from sklearn.linear_model import LogisticRegression



grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}

# C is referred to as logistic regression regularization. If C is large, it becomes overfit. If C is small, it becomes underfit. that is, we cannot model and learn data.

# l1 = lasso and l2 = ridge parameters are loss functions. You can search the web for regularization.



logreg = LogisticRegression(solver="liblinear")

logreg_cv = GridSearchCV(logreg, grid, cv = 10)

logreg_cv.fit(x_train,y_train)



print("tuned hyperparameter: (best parameters): ",logreg_cv.best_params_)

print("best accuracy according to the tuned parameter (best score): ",logreg_cv.best_score_)
logreg2 = LogisticRegression(C = 1000.0, penalty = "l2")

logreg2.fit(x_train, y_train)

print("score: ",logreg2.score(x_test, y_test))