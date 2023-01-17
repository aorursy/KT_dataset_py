from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials

from sklearn.datasets import make_classification

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from collections import Counter

import pandas as pd

import numpy as np

import pickle

import time



seed = 42 # Set seed for reproducibility purposes

metric = 'accuracy' # See other options https://scikit-learn.org/stable/modules/model_evaluation.html

kFoldSplits = 5



np.random.seed(seed) # Set numpy seed for reproducibility



# Create a toy-dataset using make_classification function from scikit-learn

X,Y=make_classification(n_samples=1000,

                        n_features=25,

                        n_informative=2,

                        n_redundant=10,

                        n_classes=2,

                        random_state=seed)



# Split in train-test-validation datasets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.25, random_state=seed) # 0.25 x 0.8 = 0.2



# Check on created data

print("Training features size:   %s x %s\nTesting features size:    %s x %s\nValidation features size: %s x %s\n" % (X_train.shape[0],X_train.shape[1], 

                                                                                                                     X_test.shape[0],X_test.shape[1], 

                                                                                                                     X_validation.shape[0],X_validation.shape[1]))



# Create a function to print variable name

def namestr(obj, namespace = globals()):

    return [name for name in namespace if namespace[name] is obj]



# Check on class distribution

for x in [Y_train, Y_test, Y_validation]:

    print(namestr(x)[0])

    counter = Counter(x)

    for k,v in counter.items():

        pct = v / len(x) * 100

        print("Class: %1.0f, Count: %3.0f, Percentage: %.1f%%" % (k,v,pct))

    print("")
space={'max_depth': hp.quniform("max_depth", 3, 18, 1),

       'gamma': hp.uniform ('gamma', 1, 9),

       'reg_alpha' : hp.quniform('reg_alpha', 40, 180, 1),

       'reg_lambda' : hp.uniform('reg_lambda', 0, 1),

       'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),

       'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),

       'n_estimators': hp.quniform('n_estimators', 50, 250, 1)}
# If regression, then: 

def hyperparameter_tuning(space):

    global best_score

    

    reg=xgb.XGBRegressor(n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],

                         reg_alpha = int(space['reg_alpha']),min_child_weight=space['min_child_weight'],

                         colsample_bytree=space['colsample_bytree'])

    

    evaluation = [(X_train, Y_train), (X_test, Y_test)]

    

    reg.fit(X_train, y_train,

            eval_set = evaluation, eval_metric = "rmse",

            early_stopping_rounds = 10,verbose = False)



    pred = reg.predict(X_test)

    mse = mean_squared_error(Y_test, pred)

    

    if (mse < best_score):

        best_score=mse

        

    # Change the metric according to the needs

    return {'loss':mse, 'status': STATUS_OK}

    

# If classifier (our case), then:

def hyperparameter_tuning(space):

    global best_score

    

    clf = XGBClassifier(n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],

                        reg_alpha = int(space['reg_alpha']),min_child_weight=space['min_child_weight'],

                        colsample_bytree=space['colsample_bytree'])

    

    evaluation = [(X_train, Y_train), (X_test, Y_test)]

    

    clf.fit(X_train, Y_train,

            eval_set = evaluation, eval_metric = 'logloss',

            early_stopping_rounds = 10, verbose = False)



    pred = clf.predict(X_test)

    accuracy = 1-accuracy_score(Y_test, pred>0.5)

    

    if (accuracy < best_score):

        best_score = accuracy

    

    # Change the metric according to the needs

    return {'loss': accuracy, 'status': STATUS_OK }
trials = Trials()

start = time.time()

neval = 100

best_score = 1.0



best = fmin(fn = hyperparameter_tuning,

            space = space,

            algo = tpe.suggest,

            max_evals = neval,

            trials = trials,

            rstate = np.random.RandomState(seed))



elapsed_time = time.time() - start
print("Parameters optimization took %.0f seconds for %d candidates. Accuracy reached: %.3f\nOptimal parameters found:\n%s" % (elapsed_time, neval, (1-best_score), best))
space={'max_depth':hp.quniform('max_depth', 1, 50, 1),

       'eta':hp.uniform ('eta', 0, 0.5),

       'subsample':hp.uniform ('subsample', 0, 1),

       'colsample_bylevel':hp.uniform ('colsample_bylevel', 0, 1),

       'colsample_bytree':hp.uniform ('colsample_bytree', 0, 1),

       'n_estimators':hp.quniform('n_estimators', 25, 500, 5),

       'gamma': hp.uniform ('gamma', 1, 25),

       'reg_alpha' : hp.quniform('reg_alpha', 25, 500, 1),

       'reg_lambda' : hp.uniform('reg_lambda', 0, 1),

       'min_child_weight' : hp.quniform('min_child_weight', 0, 50, 1)}
# Considering only our classification tuning:

def xgboost_tuning(space, kFoldSplits = 5, seed = 42, metric = 'accuracy'):

    

    global best_score, score_history, best_score_history

    

    clf = XGBClassifier(eta = space['eta'],

                        subsample = space['subsample'],

                        n_estimators = int(space['n_estimators']), 

                        max_depth = int(space['max_depth']), 

                        gamma = space['gamma'],

                        reg_alpha = int(space['reg_alpha']),

                        reg_lambda = space['reg_lambda'],

                        min_child_weight=space['min_child_weight'],

                        colsample_bytree=space['colsample_bytree'],

                        colsample_bylevel=space['colsample_bylevel'], n_jobs=-1)

    

    kfold = StratifiedKFold(n_splits=kFoldSplits, random_state=seed, shuffle=True)

    accuracy = 1-cross_val_score(clf, X_train, Y_train, cv=kfold, scoring=metric, verbose=False).mean() 

    

    if (accuracy < best_score):

        best_score = accuracy

    

    best_score_history.append(1-best_score)

    score_history.append(1-accuracy)

    

    # Change the metric according to the needs

    return {'loss': accuracy, 'status': STATUS_OK}
trials = Trials()

start = time.time()

neval = 500

best_score = 1

score_history = []

best_score_history = []



best = fmin(fn = xgboost_tuning,

            space = space,

            algo = tpe.suggest,

            max_evals = neval,

            trials = trials,

            rstate = np.random.RandomState(seed))



elapsed_time = time.time() - start
print("Parameters optimization took %.0f seconds for %d candidates. Accuracy reached: %.3f\n\nOptimal parameters found:\n%s" % (elapsed_time, neval, (1-best_score), best))
model = XGBClassifier(eta = best['eta'],

                      subsample = best['subsample'],

                      n_estimators = int(best['n_estimators']), 

                      max_depth = int(best['max_depth']), 

                      gamma = best['gamma'],

                      reg_alpha = best['reg_alpha'],

                      reg_lambda = best['reg_lambda'],

                      min_child_weight=best['min_child_weight'],

                      colsample_bytree=best['colsample_bytree'],

                      colsample_bylevel=best['colsample_bylevel'],

                      n_jobs=-1)



evaluation = [(X_train, Y_train), (X_test, Y_test)]



model.fit(X_train, Y_train,

          eval_set = evaluation, eval_metric = 'logloss',

          early_stopping_rounds = 10, verbose = False)



pred = model.predict(X_test)

accuracy = 1-accuracy_score(Y_test, pred>0.5)
print("Accuracy reached with best params on test set: %.3f" % (1-accuracy))