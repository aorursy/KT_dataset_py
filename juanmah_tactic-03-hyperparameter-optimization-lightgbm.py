import pip._internal as pip

pip.main(['install', '--upgrade', 'numpy==1.17.2'])

import numpy as np



import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV



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
lg_clf = LGBMClassifier(verbosity=VERBOSE,

                        random_state=RANDOM_STATE,

                        n_jobs=N_JOBS)
parameters = {

    'boosting_type': ['gbdt', 'dart', 'goss'] # 'rf' fails

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'num_leaves': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'max_depth': [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'n_estimators': [20, 50, 100, 200, 500, 1000, 1500, 1900, 2000, 2100, 2500]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'subsample_for_bin': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'objective': ['regression', 'binary', 'multiclass', 'lambdarank']

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'class_weight': ['balanced', None, weight]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

table_grid_search(clf, all_ranks=True)
parameters = {

    'min_split_gain': [x / 10 for x in range(0, 11)] 

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_child_weight': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_child_samples': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'subsample': [x / 10 for x in range(1, 11)]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'subsample_freq': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'colsample_bytree': [x / 10 for x in range(1, 11)]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'reg_alpha': [x / 10 for x in range(0, 11)]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'reg_lambda': [x / 10 for x in range(0, 11)]

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'importance_type': ['split', 'gain']

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
lg_clf = LGBMClassifier(verbosity=VERBOSE,

                        random_state=RANDOM_STATE,

                        n_jobs=N_JOBS)

lg_clf.min_child_weight = 1e-15

lg_clf.boosting_type = 'gbdt'

lg_clf.num_leaves = 144

lg_clf.max_depth = 21

lg_clf.learning_rate = 0.6

lg_clf.n_estimators = 2000

lg_clf.subsample_for_bin = 987

lg_clf.min_child_samples = 5

lg_clf.colsample_bytree = 0.7

lg_clf.reg_alpha = 0.7

lg_clf.reg_lambda = 0.4

parameters = {

#     'boosting_type': ['gbdt', 'goss'],

#     'num_leaves': [34, 50, 55, 60, 89],

#     'max_depth': [13, 21, 34],

#     'learning_rate': [0.5, 0.6, 0.7],

#     'n_estimators': [1900, 2000, 2100],

#     'subsample_for_bin': [610, 987, 1597],

#     'min_child_samples': [3, 5, 8, 20],

#     'colsample_bytree': [0.85, 0.9, 0.95],

#     'reg_alpha': [0.0, 0.7],

#     'reg_lambda': [0.0, 0.4]

}

# clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

# clf.fit(X_train, y_train)

# plot_grid_search(clf)

# table_grid_search(clf)
lg_clf = LGBMClassifier(verbosity=VERBOSE,

                        random_state=RANDOM_STATE,

                        n_jobs=N_JOBS)

lg_clf.min_child_weight = 1e-15

lg_clf.boosting_type = 'gbdt'

lg_clf.num_leaves = 55

lg_clf.max_depth = 21

lg_clf.learning_rate = 0.6

lg_clf.n_estimators = 2100

lg_clf.subsample_for_bin = 987

lg_clf.min_child_samples = 5

lg_clf.colsample_bytree = 0.9

lg_clf.reg_alpha = 0.0

lg_clf.reg_lambda = 0.0

parameters = {

#     'boosting_type': ['gbdt', 'goss'],

#     'num_leaves': [34, 50, 55, 60, 89],

#     'max_depth': [13, 21, 34],

#     'learning_rate': [0.5, 0.6, 0.7],

#     'n_estimators': [1900, 2000, 2100],

#     'subsample_for_bin': [610, 987, 1597],

#     'min_child_samples': [3, 5, 8, 20],

#     'colsample_bytree': [0.85, 0.9, 0.95],

#     'reg_alpha': [0.0, 0.7],

#     'reg_lambda': [0.0, 0.4]

}

# clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

# clf.fit(X_train, y_train)

# plot_grid_search(clf)

# table_grid_search(clf)
lg_clf = LGBMClassifier(verbosity=VERBOSE,

                        random_state=RANDOM_STATE,

                        n_jobs=N_JOBS)

lg_clf.min_child_weight = 1e-15

lg_clf.boosting_type = 'gbdt'

lg_clf.num_leaves = 55

lg_clf.max_depth = 21

lg_clf.learning_rate = 0.6

lg_clf.n_estimators = 2100

lg_clf.subsample_for_bin = 987

lg_clf.min_child_samples = 5

lg_clf.colsample_bytree = 0.9

lg_clf.reg_alpha = 0.0

lg_clf.reg_lambda = 0.0

# parameters = {

#     'max_depth': [15, 20, 25],

# }

parameters = {

    'boosting_type': ['gbdt', 'goss'],

    'n_estimators': [500, 1000, 1500, 2000],

    'num_leaves': [13, 21, 34, 55, 89, 144, 233, 377],

    'learning_rate': [0.5, 0.6, 0.7],

}

clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
clf.best_estimator_