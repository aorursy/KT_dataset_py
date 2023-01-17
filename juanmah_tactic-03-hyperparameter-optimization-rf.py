import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



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
rf_clf = RandomForestClassifier(verbose=VERBOSE,

                                random_state=RANDOM_STATE,

                                n_jobs=N_JOBS)
parameters = {

    'n_estimators': [200, 300, 400, 500, 550, 600, 650, 700, 750, 800, 900]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'criterion': ['gini', 'entropy']

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_depth': [10, 20, 25, 30, 35, 40, 50, 60, None]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'min_samples_split': [2, 3, 4, 5, 8, 13, 21, 34, 55, 89]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_samples_leaf': [1, 2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_weight_fraction_leaf': [x / 10 for x in range(0, 6)]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_features': ['auto', 'sqrt', 'log2', 2, 5, 8, 10, 13, 21, 34, None]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'max_leaf_nodes': [2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, None]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'min_impurity_decrease': [x / 10 for x in range(0, 11)]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'bootstrap': [True, False]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'oob_score': [True, False]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'warm_start': [True, False]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'class_weight': ['balanced', 'balanced_subsample', None]

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

# plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'n_estimators': [650, 700],

    'max_depth': [30, None],

    'max_features': [13, 21, 34, None],

    'bootstrap': [True, False],

}

clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
clf.best_estimator_