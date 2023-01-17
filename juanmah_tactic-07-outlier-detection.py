import pip._internal as pip

pip.main(['install', '--upgrade', 'numpy==1.17.2'])

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')



from lightgbm import LGBMClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier

from sklearn.ensemble import IsolationForest

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, f1_score

from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.utils.multiclass import unique_labels

from xgboost import XGBClassifier



import time

import pickle



from lwoku import RANDOM_STATE, N_JOBS, VERBOSE, get_prediction

from grid_search_utils import plot_grid_search, table_grid_search
VERBOSE=1
# Read training and test files

X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')



# Define the dependent variable

y_train = X_train['Cover_Type'].copy()



# Define a training set

X_train = X_train.drop(['Cover_Type'], axis='columns')
if_clf = IsolationForest(verbose=VERBOSE,

                         random_state=RANDOM_STATE,

                         n_jobs=N_JOBS)
f1_scorer = make_scorer(f1_score, average='micro')
parameters = {

    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800]

}

clf = GridSearchCV(if_clf, parameters, scoring=f1_scorer, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'max_samples': ['auto', 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711]

}

clf = GridSearchCV(if_clf, parameters, scoring=f1_scorer, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'contamination': [x / 20 for x in range(1, 11)]

}

clf = GridSearchCV(if_clf, parameters, scoring=f1_scorer, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'max_features': [x / 10 for x in range(1, 11)]

}

clf = GridSearchCV(if_clf, parameters, scoring=f1_scorer, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'bootstrap': [True, False]

}

clf = GridSearchCV(if_clf, parameters, scoring=f1_scorer, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'warm_start': [True, False]

}

clf = GridSearchCV(if_clf, parameters, scoring=f1_scorer, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf, all_ranks=True)
parameters = {

    'n_estimators': [100, 200],

    'max_samples': ['auto', 10000],

#     'contamination': [0.05, 0.10],

    'max_features': [0.6, 0.7, 1.0],

    'bootstrap': [True, False],

}

clf = GridSearchCV(if_clf, parameters, scoring=f1_scorer, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)

clf.fit(X_train, y_train)

plot_grid_search(clf)

table_grid_search(clf)
with open('clf.pickle', 'wb') as fp:

    pickle.dump(clf, fp)
if_clf = IsolationForest(verbose=VERBOSE,

                         random_state=RANDOM_STATE,

                         n_jobs=N_JOBS)
contamination_set = [0.01, 0.05, 0.1, 0.2]

outliers = {}

for contamination in contamination_set:

    c = str(contamination)

    print(contamination)

    if_clf.contamination = contamination

    if_clf.fit(X_train, y_train)

    outliers[c] = pd.Series(if_clf.predict(X_train), index=X_train.index)

    outliers[c].to_csv('outliers_{}.csv'.format(contamination),

                                        header=['Cover_Type'],

                                        index=True,

                                        index_label='Id')
with open('../input/tactic-03-hyperparameter-optimization-lightgbm/clf.pickle', 'rb') as fp:

    clf = pickle.load(fp)

lg_clf = clf.best_estimator_

lg_clf.class_weight = None

lg_clf
results = pd.DataFrame(columns = ['Model',

                                  'Accuracy',

                                  'Fit time',

                                  'Predict test set time',

                                  'Predict train set time'])



X_train_cleaned = {}

y_train_cleaned = {}

model = lg_clf



for contamination in contamination_set:

#     print(contamination)

    c = str(contamination)

    X_train_cleaned[c] = X_train[outliers[c] == 1]

    y_train_cleaned[c] = y_train[outliers[c] == 1]    



    # Fit

    t0 = time.time()

    model.fit(X_train_cleaned[c], y_train_cleaned[c])

    t1 = time.time()

    t_fit = (t1 - t0)

#     print(t_fit)

    

    # Predict test set

    t0 = time.time()

    y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)

    t1 = time.time()

    t_test_pred = (t1 - t0)

#     print(t_test_pred)

    

    # Predict train set

    t0 = time.time()

    y_train_pred = pd.Series(get_prediction(model, X_train_cleaned[c], y_train_cleaned[c]), index=X_train_cleaned[c].index)

    accuracy = accuracy_score(y_train_cleaned[c], y_train_pred)

    t1 = time.time()

    t_train_pred = (t1 - t0)

#     print(t_train_pred)

    

   # Submit

    y_train_pred.to_csv('train_lg_{}.csv'.format(contamination), header=['Cover_Type'], index=True, index_label='Id')

    y_test_pred.to_csv('submission_lg_{}.csv'.format(contamination), header=['Cover_Type'], index=True, index_label='Id')

    

    results = results.append({

        'Model': 'lg_cont_{}'.format(contamination),

        'Accuracy': accuracy,

        'Fit time': t_fit,

        'Predict test set time': t_test_pred,

        'Predict train set time': t_train_pred

    }, ignore_index = True)
results = results.sort_values('Accuracy', ascending=False).reset_index(drop=True)

results.to_csv('results.csv', index=True, index_label='Id')

results