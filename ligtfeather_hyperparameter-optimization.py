import pandas as pd

import numpy as np



from sklearn import ensemble

from sklearn import metrics

from sklearn import model_selection
df = pd.read_csv("../input/mobile-price-classification/train.csv")

X = df.drop("price_range", axis = 1).values #Features

y = df.price_range.values #target
classifier = ensemble.RandomForestClassifier(n_jobs=-1) #n_jobs = -1 means using all processors.

param_grid = {

    "n_estimators": [100, 200, 300, 400], 

    "max_depth": [1, 3, 7, 5],

    "criterion": ["gini", "entropy"],

}



model = model_selection.GridSearchCV(

    estimator=classifier,

    param_grid=param_grid, #Dictionary with parameters names (str) as keys and lists of parameter settings to try as values, or a list of such dictionaries

    scoring="accuracy", #A single str to evaluate the predictions on the test set.

    n_jobs=1, #Number of jobs to run in parallel

    cv=5, #Determines the cross-validation splitting strategy. If the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.

)



model.fit(X, y)
print(model.best_score_)

print(model.best_estimator_.get_params())
classifier = ensemble.RandomForestClassifier(n_jobs=-1)

param_grid = {

        "n_estimators": np.arange(100, 1500, 100),

        "max_depth": np.arange(1, 20),

        "criterion": ["gini", "entropy"],

    }

# Random search is not as expensive as grid search

model = model_selection.RandomizedSearchCV(

    estimator=classifier,

    param_distributions=param_grid,

    n_iter=10, #Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.

    scoring="accuracy",

    n_jobs=1,

    cv=5,

)

model.fit(X, y)
print(model.best_score_)    

print(model.best_estimator_.get_params())
from functools import partial

# Sequential model-based optimization in Python

from skopt import space # Initialize a search space from given specifications.

from skopt import gp_minimize # Bayesian optimization using Gaussian Processes.
# Function to minimize. Should take a single list of parameters and return the objective value.

def optimize(params, param_names, x, y):

    params = dict(zip(param_names, params)) # Create a dictonary of parameter names and values to feed into the model.

    model = ensemble.RandomForestClassifier(**params)

    kf = model_selection.KFold(n_splits=5)

    accuracies = []

    for idx in kf.split(X=x, y=y):

        train_idx, test_idx = idx[0], idx[1]

        xtrain = x[train_idx]

        ytrain = y[train_idx]



        xtest = x[test_idx]

        ytest = y[test_idx]



        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_acc = metrics.accuracy_score(ytest, preds)

        accuracies.append(fold_acc)



    return -1.0 * np.mean(accuracies)
# Initialize a search space of max_depth, n_estimators, criterion and max_features

param_space = [

    space.Integer(3, 15, name="max_depth"),

    space.Integer(100, 600, name="n_estimators"),

    space.Categorical(["gini", "entropy"], name="criterion"),

    space.Real(0.01, 1, prior = "uniform", name="max_features")

]



param_names = [

    "max_depth",

    "n_estimators",

    "criterion",

    "max_features"

]



optimization_function = partial(

    optimize,

    param_names=param_names,

    x=X,

    y=y

)



result = gp_minimize(

    optimization_function,  # Function to minimize. Should take a single list of parameters and return the objective value.

    dimensions=param_space, # List of search space dimensions.

    n_calls=15, # Number of calls to func

    n_random_starts=10, # Number of evaluations of func with random points

    verbose=10 # Control the verbosity. It is advised to set the verbosity to True for long optimization runs

)



print(dict(zip(param_names, result.x)))
from hyperopt import hp, fmin, tpe, Trials

from hyperopt.pyll.base import scope
# Optimization is finding the input value or set of values to an objective function that yields the lowest output value, called a “loss”. 

def optimize(params, x, y):

    model = ensemble.RandomForestClassifier(**params)

    kf = model_selection.KFold(n_splits=5)

    accuracies = []

    for idx in kf.split(X=x, y=y):

        train_idx, test_idx = idx[0], idx[1]

        xtrain = x[train_idx]

        ytrain = y[train_idx]



        xtest = x[test_idx]

        ytest = y[test_idx]



        model.fit(xtrain, ytrain)

        preds = model.predict(xtest)

        fold_acc = metrics.accuracy_score(ytest, preds)

        accuracies.append(fold_acc)



    return -1.0 * np.mean(accuracies)
'''There is also a few quantized versions of those functions, which rounds the generated values at each step of “q”:

    ∙ hp.quniform(label, low, high, q)

    ∙ hp.qloguniform(label, low, high, q) '''



param_space = {

    "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),

    "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),

    "criterion": hp.choice("criterion", ["gini", "entropy"]),

    "max_features": hp.uniform("max_features", 0.01, 1)

}





optimization_function = partial(

    optimize,

    x=X,

    y=y

)



trials = Trials() # It would be nice to see exactly what is happening inside the hyperopt black box. The Trials object allows us to do just that.



result = fmin(

    fn=optimization_function,

    space=param_space,

    algo=tpe.suggest,

    max_evals=15,

    trials=trials,

)



print(result)