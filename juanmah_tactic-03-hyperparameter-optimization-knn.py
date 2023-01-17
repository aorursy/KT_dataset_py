import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier



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
knn_clf = KNeighborsClassifier(n_jobs=N_JOBS)
parameters = {

    'n_neighbors': range(1, 11, 1)

}

clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'weights': ['uniform', 'distance']

}

clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

}

clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'leaf_size': range(20, 50, 5)

}

clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'p': range(1, 4)

}

clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'n_neighbors': range(1, 11, 1),

    'weights': ['uniform', 'distance'],

    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

    'leaf_size': range(20, 50, 5),

    'p': range(1, 4)

}

clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
parameters = {

    'n_neighbors': range(1, 11, 1),

    'weights': ['uniform', 'distance'],

    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],

    'p': range(1, 4)

}

clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
clf.best_estimator_