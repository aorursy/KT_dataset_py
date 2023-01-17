import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier



from lwoku import RANDOM_STATE, N_JOBS, VERBOSE, get_prediction

from grid_search_utils import plot_grid_search, table_grid_search



import pickle
# Read training and test files

X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')



# Define the dependent variable

y_train = X_train['Cover_Type'].copy()



# Define a training set

X_train = X_train.drop(['Cover_Type'], axis='columns')
VERBOSE=1
gb_clf = GradientBoostingClassifier(verbose=VERBOSE,

                                    random_state=RANDOM_STATE)
parameters = {

    'loss': ['deviance']

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'n_estimators': [100, 200, 500, 1000, 2000]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'criterion': ['friedman_mse', 'mse']

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'min_samples_split': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_samples_leaf': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_weight_fraction_leaf': [x / 10 for x in range(0, 6)]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_depth': [1, 2, 5, 8, 13, 21, 34, 53, 54, 55, 89, None]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_impurity_decrease': [x / 100 for x in range(0, 11)]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'init': ['zero', None]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

table_grid_search(clf, all_ranks=True)
parameters = {

    'max_features': ['auto', 'sqrt', 'log2', 2, 5, 8, 13, 21, 34, None]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'max_leaf_nodes': [2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, None]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'warm_start': [True, False]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'presort': [True, False, 'auto']

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'n_iter_no_change': [1],

    'validation_fraction': [x / 10 for x in range(1, 10)]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'n_iter_no_change': [1, 2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
gb_clf.subsample = 0.8

gb_clf.min_samples_split = 5

gb_clf.min_samples_leaf = 5

gb_clf.max_depth = 13

gb_clf.min_impurity_decrease = 0.03

gb_clf.max_features = 34

parameters = {

    'learning_rate': [0.01, 0.05, 0.1, 0.2],

    'n_estimators': [500, 2000],

#     'subsample': [0.8, 0.9, 1.0],

    'criterion': ['friedman_mse', 'mse'],

#     'min_samples_split': [4, 5, 6],

#     'min_samples_leaf': [4, 5, 6],

#     'max_depth': [12, 13, 14],

    'min_impurity_decrease': [0, 0.03],

#     'max_features': [21, 34, None]

}

clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
clf.best_estimator_