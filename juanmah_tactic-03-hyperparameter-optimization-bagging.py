import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier



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
bg_clf = BaggingClassifier(verbose=VERBOSE,

                           random_state=RANDOM_STATE,

                           n_jobs=N_JOBS)
parameters = {

    'n_estimators': [20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

}

clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_samples': [x / 10 for x in range(1, 11)]

}

clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.92, 0.95, 1.0]

}

clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'bootstrap': [True, False]

}

clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'bootstrap_features': [True, False]

}

clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'oob_score': [True, False]

}

clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'warm_start': [True, False]

}

clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'n_estimators': [300, 400, 500, 600, 700, 800],

    'max_features': [0.90, 0.92, 0.95, 1.0],

    'bootstrap': [True, False],

    'bootstrap_features': [True, False],

}

clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
clf.best_estimator_