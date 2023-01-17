import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
%matplotlib inline
# import and transform the data
data = pd.read_csv("../input/mushrooms.csv")
labels = data['class']
X = data.drop(['class'], axis=1)

# encode dummy variables
X = pd.get_dummies(X).values

# y should be 1 for edible
y = (labels == 'e') * 1

# split the data into train and validation
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)

# split test into test and validation
X_te, X_cv, y_te, y_cv = train_test_split(X_te, y_te, test_size=0.5, random_state=1)

print("X_tr", X_tr.shape)
print("X_cv", X_cv.shape)
print("X_te", X_te.shape)
print("y_tr", y_tr.shape)
print("y_cv", y_cv.shape)
print("y_te", y_te.shape)
# try a decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('dt', DecisionTreeClassifier())
])

grid = {
    'dt__max_depth': [5, 10, 20, 25],
}

grid_cv = GridSearchCV(pipe, grid, cv=5)

# Fit it to full training set
grid_cv.fit(X_tr, y_tr)

# Collect results and sort them
grid_results = pd.DataFrame.from_items([
    ('max_depth', grid_cv.cv_results_['param_dt__max_depth']),
    ('mean_te', grid_cv.cv_results_['mean_test_score'])
])

grid_results.sort_values(by='mean_te', ascending=False).head(10)
# score on test data
grid_cv.score(X_te, y_te)
## try a random forest
from sklearn.ensemble import RandomForestClassifier

pipe2 = Pipeline([
    ('rf', RandomForestClassifier())
])

grid2 = {
    'rf__n_estimators': [5, 10, 50],
    'rf__max_depth': [5, 10, 20, 50],
}

grid_cv2 = GridSearchCV(pipe2, grid2, cv=5)

# Fit it to full training set
grid_cv2.fit(X_tr, y_tr)

# Collect results and sort them
grid_results2 = pd.DataFrame.from_items([
    ('estimators', grid_cv2.cv_results_['param_rf__n_estimators']),
    ('max_depth', grid_cv2.cv_results_['param_rf__max_depth']),
    ('mean_te', grid_cv2.cv_results_['mean_test_score'])
])

grid_results2.sort_values(by='mean_te', ascending=False).head(10)
# score on test data
grid_cv2.score(X_te, y_te)