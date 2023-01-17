import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV



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
lda_clf = LinearDiscriminantAnalysis()
parameters = {

    'solver': ['svd', 'lsqr'] # eigen solver fails

}

clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver': ['lsqr'],

    'shrinkage': [None] + [x / 10 for x in range(0, 11)] + ['auto']

}

clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'n_components': [None] + [1, 2, 5, 8, 13, 21, 34, 55]

}

clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver': ['svd'],

    'store_covariance': [True, False]

}

clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

}

clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver': ['svd', 'lsqr'],

    'n_components': [None] + [1, 2, 5, 8, 13, 21, 34, 55],

    'store_covariance': [True, False],

    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

}

clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver': ['lsqr'],

    'shrinkage': [None] + [x / 10 for x in range(0, 11)] + ['auto'],

    'n_components': [None] + [1, 2, 5, 8, 13, 21, 34, 55],

    'store_covariance': [True, False],

    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

}

clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver': ['lsqr'],

    'shrinkage': [None] + [x / 10 for x in range(0, 11)] + ['auto']

}

clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
lda_clf = clf.best_estimator_

lda_clf