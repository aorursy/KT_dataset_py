# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.metrics import classification_report, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

import graphviz 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/pulsar_stars.csv')

df.head()

df.info()
y = df['target_class']

del df['target_class']

x = df

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
def print_scores(scores, classifier):

    print('---------------------------------')

    print(str(classifier))

    print('--------------')

    print('test score mean', scores['test_score'].mean())

    print('test score std', scores['test_score'].std())

    print('train score mean', scores['train_score'].mean())

    print('train score std', scores['train_score'].std())

    print('-----------------------------------')

    

def run_classification(clf):

    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    y_pred_proba = clf.predict_proba(x_test)

    scores = cross_validate(clf, x_train, y_train, return_train_score=True, cv=5)

    print_scores(scores, clf)

    print(classification_report(y_test, y_pred))

    print('----------------------------------')

    print('ROC AUC score', roc_auc_score(y_test, y_pred_proba[:, 1]))
dt_clf = DecisionTreeClassifier(random_state=0)

run_classification(dt_clf)

grid_clf = GridSearchCV(RandomForestClassifier(random_state=0),

                        param_grid={'n_estimators': [100, 500, 1000]}, cv=5)

grid_clf.fit(x_train, y_train)

print('Best param', grid_clf.best_params_)
run_classification(RandomForestClassifier(n_estimators=500, random_state=0))
grid_ada = GridSearchCV(AdaBoostClassifier(random_state=0),

                        param_grid={'n_estimators': [100, 500, 1000]}, cv=5)

grid_ada.fit(x_train, y_train)

print('Best param', grid_ada.best_params_)
run_classification(AdaBoostClassifier(n_estimators=500, random_state=0))
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=0),

                       param_grid={'n_estimators': [100, 500, 1000]}, cv=5)

grid_gb.fit(x_train, y_train)

print('Best param', grid_gb.best_params_)
run_classification(GradientBoostingClassifier(n_estimators=100, random_state=0))
grid_xg = GridSearchCV(XGBClassifier(random_state=0), param_grid={'n_estimators': [100, 500, 1000]}, cv=5)

grid_xg.fit(x_train, y_train)

print('Best param', grid_xg.best_params_)
run_classification(XGBClassifier(random_state=0))
