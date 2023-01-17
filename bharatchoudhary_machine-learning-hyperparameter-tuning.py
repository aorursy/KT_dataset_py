import pandas as pd
import numpy as np


from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline
if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis = 1).values
    y = df.price_range.values
    
    classifier = ensemble.RandomForestClassifier(n_jobs = -1)
    param_grid = {
        "n_esimators" : [100, 200, 300, 400],
        "max_depth" : [1,3,5,7],
        "criterion" : ["gini", "entropy"],
    }
    model = model_selection.GridSearchCV(
        estimator = classifier,
        param_grid = param_grid,
        scoring = "accuracy",
        verbose = 10,
        n_jobs = 1,
        cv = 5,
    )
    model.fit(X,y)
    
    print(model.best_score_)
    print(model.best_estimator_.get_params()) 
if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis = 1).values
    y = df.price_range.values
    
    classifier = ensemble.RandomForestClassifier(n_jobs = -1)
    param_grid = {
        "n_esimators" : np.arange(100, 1500, 100),
        "max_depth" : np.arange(1,20),
        "criterion" : ["gini", "entropy"],
    }
    model = model_selection.RandomizedSearchCV(
        estimator = classifier,
        param_distributions = param_grid,
        scoring = "accuracy",
        n_iter = 10,
        verbose = 10,
        n_jobs = 1,
        cv = 5,
    )
    model.fit(X,y)
    
    print(model.best_score_)
    print(model.best_estimator_.get_params()) 
if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis = 1).values
    y = df.price_range.values
    
    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs = -1)
    
    classifer = pipeline.Pipeline([("scaling", scl), ("pca", pca), ("rf", rf)])
    
    param_grid = {
        "pca__n_components" : np.arange(5,10),
        "rf__n_esimators" : np.arange(100, 1500, 100),
        "rf__max_depth" : np.arange(1,20),
        "rf__criterion" : ["gini", "entropy"],
    }
    model = model_selection.RandomizedSearchCV(
        estimator = classifier,
        param_distributions = param_grid,
        scoring = "accuracy",
        n_iter = 10,
        verbose = 10,
        n_jobs = 1,
        cv = 5,
    )
    model.fit(X,y)
    
    print(model.best_score_)
    print(model.best_estimator_.get_params()) 
from functools import partial
from skopt import space, gp_minimize
def optimize(params, param_names, x, y):
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X = x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        y_train = y[train_idx]
        
        xtest = x[test_idx]
        y_test = y[testin_idx]
        
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)
        
    return -1.0 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis = 1).values
    y = df.price_range.values
    
    param_space = [
        space.Integer(3, 15, name = "max_depth"),
        space.Integer(100, 600, nume = "n_estimators"),
        space.Categorical(["gini", "entropy"], name = "criterion"),
        space.Real(0.01, 1, prior = "uniform", name="max_features"),
    ]
    param_names = ["max_depth", "n_estimators", "criterion", "max_features"]
    optimization_function = partial(
        optimize,
        param_names = param_names,
        x = X,
        y = y
    )
    
    result = gp_minimize(
        optimization_function,
        dimensions = param_space,
        n_calls = 15,
        n_random_starts = 10,
        verbose = 10,
    )
    
    print(dict(zip(param_names, result.x)))
    
from functools import partial
from skopt import space, gp_minimize
from hyperopt import hp, fmin, tpe, Trails
from hyperopt.pyll.base import scope 


def optimize(params, x, y):
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X = x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        y_train = y[train_idx]
        
        xtest = x[test_idx]
        y_test = y[testin_idx]
        
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)
        
    return -1.0 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis = 1).values
    y = df.price_range.values
    
    param_space = {
        "max_depth" : scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "n_estimators" : scope.int(hp.quniform("n_estimators", 100, 600, 1)),
        "criterion" : hp.choice("criterion" , ["gini", "entropy"]),
        "max_features" : hp.uniform("max_features", 0.01, 1),
    }
    optimization_function = partial(
        optimize,
        x = X,
        y = y
    )
    
    trials = Trials()
    
    result = fmin(
        fn = optimization_function,
        space = param_space,
        algo = tpe.suggest,
        max_evals = 15,
        trials = trials,
    )
    
    print(result)
    
from functools import partial
from skopt import space, gp_minimize
from hyperopt import hp, fmin, tpe, Trails
from hyperopt.pyll.base import scope 
import optuna


def optimize(trail, x, y):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    max_depth = trail.suggest_int("max_depth", 3, 15)
    max_features = trail.suggest_uniform("max_features", 0.01, 1.0)
    
    model = ensemble.RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        max_features = max_features,
        criterion = criterion,
    )
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X = x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        y_train = y[train_idx]
        
        xtest = x[test_idx]
        y_test = y[testin_idx]
        
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)
        
    return -1.0 * np.mean(accuracies)

if __name__ == "__main__":
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop("price_range", axis = 1).values
    y = df.price_range.values
    optimization_function = partial(optimize, x = X, y = y)
    
    study = optuna.create_study(direction= "minimize")
    study.optimize(optimization_function, n_trials = 15)
    
    
