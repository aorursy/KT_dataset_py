# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/voice.csv')
data.head()
data.info()
data.describe()
# feature names as a list

col = data.columns       # .columns gives columns names in data 

print(col)
sns.countplot(x="label", data=data)

data.loc[:,'label'].value_counts()
y = data.label.values

x_data = data.drop(["label"],axis=1)
# Normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#  train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3)
# knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)  # k = n_neighbors
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = knn, X = x_train, y= y_train, cv = 10)

print("average accuracy: ",np.mean(accuracies))

print("average std: ",np.std(accuracies))
knn.fit(x_train,y_train)

print("test accuracy: ",knn.score(x_test,y_test))
# grid search cross validation for knn



from sklearn.model_selection import GridSearchCV



grid = {"n_neighbors":np.arange(1,50)}

knn= KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv = 10)  # GridSearchCV

knn_cv.fit(x,y)
# print hyperparameter KNN algoritmasindaki K degeri

print("Tuned hyperparameter K: ",knn_cv.best_params_)

print("Best accuracy for tuned parameter (best score): ",knn_cv.best_score_)
# Grid search CV with logistic regression



from sklearn.linear_model import LogisticRegression
grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg,grid,cv = 10)

logreg_cv.fit(x,y)
print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)

print("accuracy: ",logreg_cv.best_score_)
