# pip install hyperopt
from sklearn.datasets import make_classification

import pandas as pd

import numpy as np



seed = 42 # Set seed for reproducibility purposes

metric = 'accuracy' # See other options https://scikit-learn.org/stable/modules/model_evaluation.html

kFoldSplits = 5



np.random.seed(seed)

# random.seed(seed) # If importing random, should also set this seed

# tf.random.set_seed(seed) # If using tensorflow, should also set this seed



X,Y=make_classification(n_samples=500,

                        n_features=30,

                        n_informative=2,

                        n_redundant=10,

                        n_classes=2,

                        random_state=seed)
# If kFold is not defined then 5-fold default value was used.

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC

import time



classifier = SVC()



#Parameter search space for both GridSearch and RandomizedSearch

params={

        'C': np.arange(0.005,1.0,0.01),

        'kernel': ['linear', 'poly', 'rbf'],

        'degree': [2,3,4],

        'probability':[True]

        }



n_iter = 100

random_search = RandomizedSearchCV(classifier, 

                                   param_distributions = params,

                                   n_iter = n_iter,

                                   scoring = metric,

                                   random_state=seed,

                                   cv = StratifiedKFold(n_splits=kFoldSplits, random_state=seed, shuffle=True).split(X,Y))



# Run the fit and time it

start = time.time()

random_search.fit(X,Y)

elapsed_time_random = time.time() - start



grid_search = GridSearchCV(classifier,

                           param_grid = params,

                           scoring = metric,

                           cv = StratifiedKFold(n_splits=kFoldSplits, random_state=seed, shuffle=True).split(X,Y))



start = time.time()

grid_search.fit(X,Y)

elapsed_time_grid = time.time() - start



best_score=1.0



def objective(space):

    

    global best_score

    model = SVC(**space)   

    kfold = StratifiedKFold(n_splits=kFoldSplits, random_state=seed, shuffle=True) # KFold is also an option.

    score = 1-cross_val_score(model, X, Y, cv=kfold, scoring=metric, verbose=False).mean() 

    # Careful here (score). The objective function will be  minimized, thus somme treatment on your score might be needed.

    

    if (score < best_score):

        best_score=score

    

    return score 
from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials



space = {

      'C': hp.choice('C', np.arange(0.005,1.0,0.01)),

      'kernel': hp.choice('kernel',['linear', 'poly', 'rbf']),

      'degree':hp.choice('degree',[2,3,4]),

      'probability':hp.choice('probability',[True])

      }
n_iter_hopt = 50

trials = Trials() # Initialize an empty trials database for further saving/loading ran iteractions



start = time.time()



best = fmin(objective, 

            space = space, 

            algo = tpe.suggest, 

            max_evals = n_iter_hopt,

            trials = trials,

            rstate = np.random.RandomState(seed))



elapsed_time_hopt = time.time() - start
print("\nGridSearchCV took %.0f seconds for all candidates. Accuracy reached: %.3f\nOptimal parameters found: %s" % (elapsed_time_grid, ((grid_search.best_score_)*100), grid_search.best_params_))

print("\nRandomizedSearchCV took %.0f seconds for %d candidates. Accuracy reached: %.3f\nOptimal parameters found: %s" % (elapsed_time_random, n_iter, ((random_search.best_score_)*100), random_search.best_params_))

print("\nHyperopt search took %.2f seconds for %d candidates. Accuracy reached: %.3f\nOptimal parameters found: %s" % (elapsed_time_hopt, n_iter_hopt, ((1-best_score)*100), best))
# More details on how to work with save/load trials in https://github.com/hyperopt/hyperopt/issues/267

import pickle

pickle.dump(trials, open("trials.p", "wb"))

trials = pickle.load(open("trials.p", "rb")) # Pass this to Hyperopt during the next training run.
from xgboost import XGBClassifier



n_iter_hopt_xgb = 50

trials = Trials() # Initialize an empty trials database for further saving/loading ran iteractions



# Declare xgboost search space for Hyperopt

xgboost_space={

            'max_depth': hp.choice('x_max_depth',[2,3,4,5,6]),

            'min_child_weight':hp.choice('x_min_child_weight',np.round(np.arange(0.0,0.2,0.01),5)),

            'learning_rate':hp.choice('x_learning_rate',np.round(np.arange(0.005,0.3,0.01),5)),

            'subsample':hp.choice('x_subsample',np.round(np.arange(0.1,1.0,0.05),5)),

            'colsample_bylevel':hp.choice('x_colsample_bylevel',np.round(np.arange(0.1,1.0,0.05),5)),

            'colsample_bytree':hp.choice('x_colsample_bytree',np.round(np.arange(0.1,1.0,0.05),5)),

            'n_estimators':hp.choice('x_n_estimators',np.arange(25,100,5))

            }



best_score_xgb = 1.0



def objective(space):

    

    global best_score_xgb

    model = XGBClassifier(**space, n_jobs=-1)   

    kfold = StratifiedKFold(n_splits=kFoldSplits, random_state=seed, shuffle=True)

    score = 1-cross_val_score(model, X, Y, cv=kfold, scoring=metric, verbose=False).mean() 

    

    if (score < best_score_xgb):

        best_score_xgb=score

    

    return score 



start = time.time()



best = fmin(objective, 

            space = xgboost_space, 

            algo = tpe.suggest, 

            max_evals = n_iter_hopt_xgb,

            trials = trials,

            rstate = np.random.RandomState(seed))



elapsed_time_xgb = (time.time() - start)
print("Hyperopt on XGBoost took %.0f seconds for %d candidates. Accuracy reached: %.3f\nOptimal parameters found: %s" % (elapsed_time_xgb, n_iter_hopt_xgb, ((1-best_score_xgb)*100), best))
import matplotlib.pyplot as plt



score_history = []

best_score_history = []

trials = Trials()

n_iter_hopt_xgb = 200



best_score_xgb = 1.0



def objective(space):

    

    global best_score_xgb

    model = XGBClassifier(**space, n_jobs=-1)   

    kfold = StratifiedKFold(n_splits=kFoldSplits, random_state=seed, shuffle=True)

    score = 1-cross_val_score(model, X, Y, cv=kfold, scoring=metric, verbose=False).mean() 

    

    if (score < best_score_xgb):

        best_score_xgb=score

    

    # To visualize, we need to modify our objective function to capture history

    best_score_history.append(1-best_score)

    score_history.append(1-score)

    

    return score 



start = time.time()



best = fmin(objective, 

            space = xgboost_space, 

            algo = tpe.suggest, 

            max_evals = n_iter_hopt_xgb,

            trials = trials,

            rstate = np.random.RandomState(seed))



elapsed_time_xgb = (time.time() - start)
plotY = score_history

plotX = list(range(1, n_iter_hopt_xgb+1, 1))

plt.figure(figsize=(10,8))

plt.xlabel('Iteration')

plt.ylabel('Accuracy')

plt.title('Hyperopt Search Pattern')

plt.plot(plotX, plotY, 'ro')  
plt.figure(figsize = (10,8))

plt.xlabel('Accuracy')

plt.ylabel('Frequency')

plt.title('Histogram of Hyperopt Solution Scores')

plt.hist(plotY, 30, density=True, facecolor='r', alpha=0.75)

plt.show