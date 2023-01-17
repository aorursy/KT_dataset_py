# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python 

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import average_precision_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

import joblib

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/ccfraudsetup/train.csv',index_col=0)

test=pd.read_csv('../input/ccfraudsetup/test.csv',index_col=0)

X_train = train.iloc[:,  :-1]

y_train = train.iloc[:,-1:  ]

X_test = test.iloc[:,  :-1]

y_test = test.iloc[:,-1:  ]
space = {'params': {'max_depth': hp.uniformint('max_depth', 22, 44),

                    'max_features': hp.uniformint('max_features', 21, 29),

                    'max_leaf_nodes': hp.uniformint('max_leaf_nodes', 12, 35),

                    'min_samples_leaf': hp.uniformint('min_samples_leaf', 5, 29),

                    'min_samples_split': hp.uniformint('min_samples_split', 43, 68),

                    'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 5.87E-07, 2.04E-04),

                    'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),

                    'n_estimators': 300,

                    'n_jobs': -1

                    }}





def objective(search_instance):

    params = search_instance['params']

    clf = RandomForestClassifier(**params)



    X_trn = X_train

    y_trn = y_train

    X_tst = X_test

    y_tst = y_test



    score = (cross_val_score(clf, X_trn, y_trn.values.ravel(), scoring='average_precision', cv=3, verbose=2, n_jobs=-1))

    overfit_score = np.min(score)

    print('cv: ', score, 'using: ',  overfit_score)



    return {'loss': 1 - overfit_score, 'params': params, 'status': STATUS_OK}

try:

    joblib.load('trials.joblib')

except:

    print( 'error')

    trials = Trials()


best = fmin(objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)



joblib.dump(trials, 'trials.joblib')

pd.DataFrame(trials).to_csv('trials.csv')