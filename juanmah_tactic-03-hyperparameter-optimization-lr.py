import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LogisticRegression

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
lr_clf = LogisticRegression(verbose=VERBOSE,

                            random_state=RANDOM_STATE,

                            n_jobs=N_JOBS)
parameters = {

    'solver': ['newton-cg', 'sag', 'lbfgs'],

    'penalty': ['none', 'l2']

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'solver': ['liblinear'],

    'penalty': ['l1', 'l2']

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'solver': ['saga'],

    'l1_ratio': [x / 10 for x in range(0, 11)],

    'penalty': ['none', 'elasticnet']

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver': ['liblinear'],

    'penalty': ['l2'],

    'dual': [True, False]

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'C': [(0.9 + x / 50) for x in range(0, 10)]

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'fit_intercept': [True, False]

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver' : ['liblinear'],

    'fit_intercept': [True],

    'intercept_scaling': [1, 2, 3, 5, 8, 13, 21, 34]

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'class_weight' : [None, 'balanced']

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

table_grid_search(clf, all_ranks=True)
parameters = {

    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'max_iter': range(50, 250, 50)

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],

    'multi_class': ['ovr', 'multinomial']

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],

    'warm_start': [True, False]

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'solver': ['liblinear'],

    'penalty': ['l1', 'l2'],

    'C': [0.98, 1.00, 1.02],

    'tol': [1e-7, 1e-8, 1e-9],

    'max_iter': range(100, 250, 50)

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf_liblinear.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
parameters = {

    'solver': ['saga'],

    'max_iter': range(100, 250, 50)

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf_saga.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
parameters = {

    'solver': ['sag'],

    'max_iter': range(100, 250, 50),

    'multi_class': ['ovr', 'multinomial'],

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf_sag.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
parameters = {

    'solver': ['lbfgs'],

    'penalty': ['none', 'l2'],

    'C': [0.98, 1.00, 1.02],

    'fit_intercept': [True, False],

    'max_iter': range(100, 250, 50),

    'multi_class': ['ovr', 'multinomial'],

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf_lbfgs.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
parameters = {

    'solver': ['newton-cg'],

    'penalty': ['none', 'l2'],

    'C': [0.98, 1.00, 1.02],

    'fit_intercept': [True, False],

    'max_iter': range(100, 250, 50),

    'multi_class': ['ovr', 'multinomial'],

}

clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf_newton-cg.pickle', 'wb') as fp:

    pickle.dump(clf, fp)