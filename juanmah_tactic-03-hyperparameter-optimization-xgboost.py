import pip._internal as pip

pip.main(['install', '--upgrade', 'numpy==1.17.2'])

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier



from lwoku import RANDOM_STATE, N_JOBS, VERBOSE, get_prediction

from grid_search_utils import plot_grid_search, table_grid_search



import pickle
VERBOSE=1
# Read training and test files

X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')



# Define the dependent variable

y_train = X_train['Cover_Type'].copy()



# Define a training set

X_train = X_train.drop(['Cover_Type'], axis='columns')
xg_clf = XGBClassifier(verbosity=VERBOSE,

                       random_state=RANDOM_STATE,

                       n_jobs=N_JOBS)
parameters = {

    'max_depth': [1, 2, 3, 5, 8, 13, 21, 34, 55]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'n_estimators': [20, 50, 100, 200, 500, 1000, 1500, 2000, 2500]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'booster': ['gbtree', 'gblinear', 'dart']

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'gamma': [0, 1, 2, 3, 5, 8]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_child_weight': [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_delta_step': [0, 1, 2, 3, 5, 8]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_delta_step': [x / 10 for x in range(1, 11)]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'colsample_bytree': [x / 10 for x in range(0, 11)]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'colsample_bylevel': [x / 10 for x in range(0, 11)]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'colsample_bynode': [x / 10 for x in range(0, 11)]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'reg_alpha': [x / 10 for x in range(0, 11)]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'reg_lambda': [x / 10 for x in range(0, 11)]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'scale_pos_weight': [-0.5 + x / 10 for x in range(0, 11)]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'importance_type': ['gain', 'weight', 'cover', 'total_gain', 'total_cover']

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
xg_clf = XGBClassifier(verbosity=VERBOSE,

                       random_state=RANDOM_STATE,

                       n_jobs=N_JOBS)

xg_clf.n_estimators = 500

xg_clf.max_depth = 21

xg_clf.learning_rate = 1

xg_clf.gamma = 2

xg_clf.min_child_weight = 2

xg_clf.max_delta_step = 2

xg_clf.subsample = 0.7

xg_clf.colsample_bytree = 0.5

xg_clf.colsample_bylevel = 0.4

xg_clf.colsample_bynode = 0.4

xg_clf.reg_lambda = 0

parameters = {

    'n_estimators': [500],

    'learning_rate': [0.1, 1],

    'gamma': [0, 2],

#     'min_child_weight': [1, 2],

#     'max_delta_step': [0, 2],

#     'subsample': [0.7, 1],

#     'colsample_bytree': [0.5, 1],

#     'colsample_bylevel': [0.4, 1],

#     'colsample_bynode': [0.4, 1],

#     'reg_lambda': [0, 1]

}

clf = GridSearchCV(xg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
clf.best_estimator_