# import pip._internal as pip

# pip.main(['install', '--upgrade', 'numpy==1.17.2'])
!pip install --upgrade numpy==1.17.2
import numpy as np

print('> NumPy version: {}'.format(np.__version__))
import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import time

from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_predict
print('> Set model parameters')

N_ESTIMATORS = 200

MIN_SAMPLE_LEAFS = 50

RANDOM_STATE = 1

N_JOBS = -1

VERBOSE = 0
print('> Load functions')



print('  -- Get prediction')

# noinspection PyPep8Naming

def get_prediction(estimator, X, y):

    y_pred = cross_val_predict(estimator, X, y, cv=5, n_jobs=N_JOBS)

    return y_pred
print('> Prepare data')



print('  -- Read training and test files')

X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')



print('  -- Define the dependent variable')

y_train = X_train['Cover_Type'].copy()



print('  -- Define a training set')

X_train = X_train.drop(['Cover_Type'], axis='columns')
print('> Define Model')



print('  -- LightGBM')

lg_clf = LGBMClassifier(n_estimators=N_ESTIMATORS,

                        num_leaves=MIN_SAMPLE_LEAFS,

                        verbosity=VERBOSE,

                        random_state=RANDOM_STATE,

                        n_jobs=N_JOBS)
print('> Fit')

t0 = time.time()

lg_clf.fit(X_train, y_train)

t1 = time.time()

t_fit = (t1 - t0)

print(t_fit)
print('> Predict')

print('  -- test set')

t0 = time.time()

y_test_pred = pd.Series(lg_clf.predict(X_test), index=X_test.index)

t1 = time.time()

t_test_pred = (t1 - t0)

print(t_test_pred)



print('  -- train set')

t0 = time.time()

y_train_pred = get_prediction(lg_clf, X_train, y_train)

accuracy = accuracy_score(y_train, y_train_pred)

t1 = time.time()

t_train_pred = (t1 - t0)

print(t_train_pred)
print('> Export results')

results = pd.DataFrame(columns = ['Model',

                                  'Accuracy',

                                  'Fit time',

                                  'Predict test set time',

                                  'Predict train set time'])

results = results.append({

    'Model': 'LightGBM',

    'Accuracy': accuracy,

    'Fit time': t_fit,

    'Predict test set time': t_test_pred,

    'Predict train set time': t_train_pred

}, ignore_index = True)



print(results)