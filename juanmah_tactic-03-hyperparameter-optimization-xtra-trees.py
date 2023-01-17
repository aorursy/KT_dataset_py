import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier



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
xt_clf = ExtraTreesClassifier(verbose=VERBOSE,

                              random_state=RANDOM_STATE,

                              n_jobs=N_JOBS)
parameters = {

    'n_estimators': [10, 20, 50, 100, 200, 500, 1000, 1200, 1500, 1800, 1900, 2000, 2100, 3000]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'criterion': ['gini', 'entropy']

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_depth': [1, 2, 5, 8, 13, 21, 34, 53, 54, 55, 89, None]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_samples_split': [2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_samples_leaf': [1, 2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_weight_fraction_leaf': [x / 10 for x in range(0, 6)]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_features': ['auto', 'sqrt', 'log2', 2, 5, 8, 13, 21, 34, None]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'max_leaf_nodes': [2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, None]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_impurity_decrease': [x / 100 for x in range(0, 11)]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'bootstrap': [True, False]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'bootstrap': [True],

    'oob_score': [True, False]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'warm_start': [True, False]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'class_weight': ['balanced', 'balanced_subsample', None]

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
xt_clf.max_features = None

parameters = {

    'n_estimators': range(1800, 2100, 10)

}

clf = GridSearchCV(xt_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
clf.best_estimator_