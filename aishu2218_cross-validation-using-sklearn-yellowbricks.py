# Import required libraries

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import sklearn

import warnings

warnings.filterwarnings('ignore')
# Import necessary modules

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import LeavePOut

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import cv_scores

from yellowbrick.model_selection import CVScores

data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.head()
x1 = data.drop('Outcome', axis=1).values 

y1 = data['Outcome'].values
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)

model = LogisticRegression()

model.fit(X_train, Y_train)

result = model.score(X_test, Y_test)

print("Accuracy: %.2f%%" % (result*100.0))
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=22)

model = LogisticRegression()

model.fit(X_train, Y_train)

result = model.score(X_test, Y_test)

print("Accuracy: %.2f%%" % (result*100.0))
loocv = LeaveOneOut()

model_loocv = LogisticRegression()

results_loocv = cross_val_score(model_loocv, x1, y1, cv=loocv)

print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))
for k in range(2,11):



    kfold = KFold(n_splits=k, random_state=100)

    model_kfold = LogisticRegression()

    results_kfold = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold)

    print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0)) 
# Create a cross-validation strategy

cv = KFold(n_splits=10, random_state=100)



# Instantiate the classification model and visualizer

model = LogisticRegression()



# Fit the data to the visualizer

visualizer = cv_scores(model, x1, y1, cv=cv, scoring='accuracy')
for k in range(2,11):



    skfold = StratifiedKFold(n_splits=k, random_state=100)

    model_skfold = LogisticRegression()

    results_skfold = cross_val_score(model_skfold, x1, y1, cv=skfold)



    print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
# Create a cross-validation strategy

cv = StratifiedKFold(n_splits=10, random_state=100)



# Instantiate the classification model and visualizer

model = LogisticRegression()

visualizer = CVScores(model, cv=cv, scoring='accuracy')



# Fit the data to the visualizer

visualizer.fit(x1, y1)



# Finalize and render the figure

visualizer.show()           
for k in range(2,11):



    kfold2 = ShuffleSplit(n_splits=k, test_size=0.30, random_state=100)

    model_shufflecv = LogisticRegression()

    results_4 = model_selection.cross_val_score(model_shufflecv, x1, y1, cv=kfold2)

    print("Accuracy: %.2f%% (%.2f%%)" % (results_4.mean()*100.0, results_4.std()*100.0))
# Create a cross-validation strategy

cv = ShuffleSplit(n_splits=10, random_state=100)



# Instantiate the classification model and visualizer

model = LogisticRegression()

visualizer = CVScores(model, cv=cv, scoring='accuracy')



# Fit the data to the visualizer

visualizer.fit(x1, y1)



# Finalize and render the figure

visualizer.show()  