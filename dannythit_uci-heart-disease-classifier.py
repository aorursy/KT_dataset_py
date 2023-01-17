import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
df.describe()
df.info()
sns.heatmap(df.corr())
df['cp'].value_counts()
df.head()
features = df.iloc[: , :-1]

target = df['target']
features.dtypes
from sklearn.model_selection import train_test_split, GridSearchCV



X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
from sklearn import tree

from sklearn.metrics import f1_score



clf_tree = tree.DecisionTreeClassifier(criterion='entropy',random_state=42)

clf_tree.fit(X_train, y_train)



tree_y_pred = clf_tree.predict(X_test)



f1_score(y_test, tree_y_pred)
clf_tree = GridSearchCV(clf_tree , {

    'criterion': ['entropy', 'gini'],

    'max_depth': [2, 5, 10, 15],

    'min_samples_split': [2, 3, 5, 7],

    'max_features': [2, 4, 6, 8]

}, return_train_score=False)



clf_tree.fit(features, target)

clf_tree.cv_results_
data = pd.DataFrame(clf_tree.cv_results_)
data.head()
final_data = data[['param_max_depth', 'param_max_features', 'param_min_samples_split', 'mean_test_score']]
final_data.head()
final_data.nlargest(5, 'mean_test_score')
# The best params for decision tree

clf_tree.best_params_
# The best score for decision tree 

clf_tree.best_score_
from sklearn.linear_model import LogisticRegression



clf_lr = LogisticRegression(random_state=42, max_iter=500)
clf_lr = clf_lr.fit(X_train, y_train)



lr_y_pred = clf_lr.predict(X_test)



f1_score(y_test, lr_y_pred)



# Logistic Regression provided 82% accuracy (best one so far)
from sklearn.ensemble import RandomForestClassifier



clf_rf = RandomForestClassifier(random_state=42)



clf_rf = clf_rf.fit(X_train, y_train)



rf_y_pred = clf_rf.predict(X_test)



f1_score(y_test, rf_y_pred)



# Random Forest provided 86% accuracy without tuning any parameters (best one)
param_grid = { 

    'n_estimators': [100, 200],

    'max_depth' : [3,4],

    'criterion' :['gini', 'entropy'],

    'min_samples_leaf' : [1,2,3],

    'min_samples_split' : [2,3]

}



clf_rfc = RandomForestClassifier(random_state=42)



clf_rfc = GridSearchCV(estimator = clf_rfc , param_grid = param_grid, cv=3, return_train_score=False)
clf_rfc.fit(X_train, y_train)
clf_rfc.cv_results_
rfc_df = pd.DataFrame(clf_rfc.cv_results_)
rfc_df.head()
clf_rfc.best_params_
clf_rfc.best_score_