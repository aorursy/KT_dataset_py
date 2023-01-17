# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import ensemble

from sklearn import model_selection

from sklearn import metrics

from sklearn import preprocessing, decomposition, pipeline



from skopt import space, gp_minimize

from functools import partial



from hyperopt import hp, fmin, tpe, Trials

from hyperopt.pyll.base import scope



import optuna
path = '/kaggle/input/mobile-price-classification'

df = pd.read_csv(os.path.join(path, 'train.csv'))

df.head()
df.shape
df['price_range'].value_counts()
X = df.drop(['price_range'], axis = 1).values

y = df['price_range'].values
y.shape
classifier = ensemble.RandomForestClassifier(n_jobs = -1)

param_grid = {

    'n_estimators': [100, 200, 300, 400, 500],

    'max_depth' : [1, 3, 5, 7],

    'criterion':['gini','entropy']

}



model = model_selection.GridSearchCV(

        estimator=classifier,

        param_grid=param_grid,

        scoring='accuracy',

        n_jobs=1,

        verbose = 10,

        cv = 5)

model.fit(X, y)
print(model.best_score_)

print(model.best_estimator_.get_params())
param_grid = {

    'n_estimators':np.arange(100, 1500, 100),

    'max_depth':np.arange(1, 20),

    'criterion': ['gini', 'entropy']

}



model = model_selection.RandomizedSearchCV(estimator=classifier,

                                          param_distributions=param_grid,

                                          n_iter = 10,

                                          scoring='accuracy',

                                          verbose = 10,

                                          n_jobs = 1,

                                          cv = 5)
model.fit(X, y)

print(model.best_score_)

print(model.best_estimator_.get_params())
scl = preprocessing.StandardScaler()

dec = decomposition.PCA()

rf = ensemble.RandomForestClassifier(n_jobs = -1)



classifier = pipeline.Pipeline(

                                [

                                    ('scaling',scl),

                                    ('decomposition',dec),

                                    ('rf',rf)

                                ])
param_grid = {

    'decomposition__n_components':np.arange(5, 10),

    'rf__n_estimators':np.arange(100, 1500, 100),

    'rf__max_depth':np.arange(1, 20),

    'rf__criterion': ['gini', 'entropy']

}
model = model_selection.RandomizedSearchCV(estimator=classifier,

                                          param_distributions=param_grid,

                                          n_iter = 10,

                                          scoring='accuracy',

                                          verbose = 10,

                                          n_jobs = 1,

                                          cv = 5)

model.fit(X, y)
print(model.best_score_)

print(model.best_estimator_.get_params())
def optimize(params, param_names, x, y):

    params = dict(zip(param_names, params))

    model = ensemble.RandomForestClassifier(**params)

    kf = model_selection.StratifiedKFold(n_splits = 5)

    accuracies = []

    for idx in kf.split(X =x, y=y):

        train_idx, test_idx = idx[0], idx[1]

        xtrain = x[train_idx]

        ytrain = y[train_idx]

        

        xtest = x[test_idx]

        ytest = y[test_idx]

        

        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_acc = metrics.accuracy_score(ytest, preds)

        accuracies.append(fold_acc)

        

    return -1.0 *np.mean(accuracies)

        
param_space = [

    space.Integer(3, 15, name = 'max_depth'),

    space.Integer(100, 600, name = 'n_estimators'),

    space.Categorical(['gini', 'entropy'], name = 'criterion'),

    space.Real(0.01, 1, prior = 'uniorm', name = 'max_features')

]



param_names = [

    'max_depth',

    'n_estimators',

    'criterion',

    'max_features'

]
optimization_function = partial(optimize, 

                                param_names  =param_names,

                                x = X,

                                y = y

)
result = gp_minimize(optimization_function,

                    dimensions=param_space,

                    n_calls=15,

                    n_random_starts=10,

                    verbose = 10)
print(dict(zip(param_names, result.x)))
def optimize(params, x, y):

    model = ensemble.RandomForestClassifier(**params)

    kf = model_selection.StratifiedKFold(n_splits = 5)

    accuracies = []

    for idx in kf.split(X =x, y=y):

        train_idx, test_idx = idx[0], idx[1]

        xtrain = x[train_idx]

        ytrain = y[train_idx]

        

        xtest = x[test_idx]

        ytest = y[test_idx]

        

        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_acc = metrics.accuracy_score(ytest, preds)

        accuracies.append(fold_acc)

        

    return -1.0 *np.mean(accuracies)



param_space = {

    'max_depth': scope.int(hp.quniform("max_depth", 3, 15,1)),

    'n_estimators': scope.int(hp.quniform("n_estimators", 100, 600,1)),

    'criterion': hp.choice("criterion", ['gini', 'entropy']),

    'max_features': hp.uniform("max_features", 0.01, 1)

}

        

    

optimization_function = partial(optimize, x = X, y= y)



trials = Trials()



result = fmin(fn=optimization_function,

             space = param_space,

             algo = tpe.suggest,

             max_evals= 15,

             trials = trials,

             verbose = 10)
print(result)
def optimize(trial, x, y):

    

    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    n_estimators = trial.suggest_int("n_estimators", 100, 1500)

    max_depth = trial.suggest_int("max_depth", 3, 15)

    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)

    

    model = ensemble.RandomForestClassifier(criterion=criterion,

                                           n_estimators = n_estimators,

                                           max_depth = max_depth,

                                           max_features = max_features)

    

    kf = model_selection.StratifiedKFold(n_splits = 5)

    accuracies = []

    for idx in kf.split(X =x, y=y):

        train_idx, test_idx = idx[0], idx[1]

        xtrain = x[train_idx]

        ytrain = y[train_idx]

        

        xtest = x[test_idx]

        ytest = y[test_idx]

        

        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_acc = metrics.accuracy_score(ytest, preds)

        accuracies.append(fold_acc)

        

    return -1.0 *np.mean(accuracies)
optimization_function = partial(optimize, x=X, y= y)

study = optuna.create_study(direction = 'minimize')

study.optimize(optimization_function, n_trials = 15)
