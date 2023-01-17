# from sklearn.model_selection import GridSearchCV

from ray.tune.sklearn import TuneGridSearchCV



from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.datasets import load_iris

import numpy as np





iris = load_iris()

X = iris.data

y = iris.target



x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 14)



# Example parameters to tune from SGDClassifier

parameter_grid = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}



tune_search = TuneGridSearchCV(SGDClassifier(),parameter_grid,early_stopping=True,max_iters=10)



tune_search.fit(x_train, y_train)

print(tune_search.best_score)

print(tune_search.best_params_)
# !pip install optuna
import optuna

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold , cross_val_score

from sklearn.datasets import load_iris



iris = load_iris()

X = iris.data

y = iris.target



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 14)



def objective(trial):



    optimizer = trial.suggest_categorical('algorithm', ['auto','ball_tree','kd_tree','brute'])

    rf_max_depth = trial.suggest_int("k_n_neighbors", 2, 10, log=True)

    knn = KNeighborsClassifier(n_neighbors=rf_max_depth,algorithm=optimizer)



    score = cross_val_score(knn, X_train,y_train, n_jobs=-1, cv=3)

    accuracy = score.mean()

    return accuracy





if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=10)

    print(study.best_trial)

    

#best parameter combination

study.best_params



#score achieved with best parameter combination

study.best_value
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials , space_eval

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_iris



iris = load_iris()

X = iris.data

y = iris.target



def hyperopt_train_test(params):

    clf = RandomForestClassifier(**params)

    return cross_val_score(clf, X, y).mean()



space = {

    'max_depth': hp.choice('max_depth', range(1,20)),

    'max_features': hp.choice('max_features', range(1,5)),

    'n_estimators': hp.choice('n_estimators', range(1,20)),

    'criterion': hp.choice('criterion', ["gini", "entropy"])

            }

best = 0

def f(params):

    global best

    acc = hyperopt_train_test(params)

    if acc > best:

      best = acc

      print( 'new best:', best, params)

    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()

best = fmin(f, space, algo=tpe.suggest, max_evals=300, trials=trials)

print(best)
# !pip install parameter-sherpa
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

import time

import sherpa

import sherpa.algorithms.bayesian_optimization as bayesian_optimization





parameters = [sherpa.Discrete('n_estimators', [2, 50]),

              sherpa.Choice('criterion', ['gini', 'entropy']),

              sherpa.Continuous('max_features', [0.1, 0.9])]



algorithm = bayesian_optimization.GPyOpt(max_concurrent=1,

                                         model_type='GP_MCMC',

                                         acquisition_type='EI_MCMC',

                                         max_num_trials=10)



X, y = load_breast_cancer(return_X_y=True)

study = sherpa.Study(parameters=parameters,

                     algorithm=algorithm,

                     lower_is_better=False)



for trial in study:

    print("Trial ", trial.id, " with parameters ", trial.parameters)

    clf = RandomForestClassifier(criterion=trial.parameters['criterion'],

                                 max_features=trial.parameters['max_features'],

                                 n_estimators=trial.parameters['n_estimators'],

                                 random_state=0)

    scores = cross_val_score(clf, X, y, cv=5)

    print("Score: ", scores.mean())

    study.add_observation(trial, iteration=1, objective=scores.mean())

    study.finalize(trial)

print(study.get_best_result())
# !pip install scikit-optimize
from skopt import BayesSearchCV



import warnings

warnings.filterwarnings("ignore")



# parameter ranges are specified by one of below

from skopt.space import Real, Categorical, Integer



knn = KNeighborsClassifier()

#defining hyper-parameter grid

grid_param = { 'n_neighbors' : list(range(2,11)) , 

              'algorithm' : ['auto','ball_tree','kd_tree','brute'] }



#initializing Bayesian Search

Bayes = BayesSearchCV(knn , grid_param , n_iter=30 , random_state=14)

Bayes.fit(X_train,y_train)



#best parameter combination

print(f'Best parameter combination : {Bayes.best_params_}')



#score achieved with best parameter combination

print(f'Best Score : {Bayes.best_score_}')



#all combinations of hyperparameters

# print(Bayes.cv_results_['params'])



#average scores of cross-validation

# Bayes.cv_results_['mean_test_score']
!pip install gpyopt
import GPy

import GPyOpt

from GPyOpt.methods import BayesianOptimization

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.datasets import load_iris

from scipy.stats import uniform

from xgboost import XGBRegressor

import numpy as np



iris = load_iris()

X = iris.data

y = iris.target



x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 14)



bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},

        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},

        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)}]



# Optimization objective 

def cv_score(parameters):

    parameters = parameters[0]

    score = cross_val_score(

                XGBRegressor(learning_rate=parameters[0],

                              gamma=int(parameters[1]),

                              max_depth=int(parameters[2])), 

                X, y, scoring='neg_mean_squared_error').mean()

    score = np.array(score)

    return score



optimizer = GPyOpt.methods.BayesianOptimization(f = cv_score,            # function to optimize       

                                          domain = bds,         # box-constraints of the problem

                                          acquisition_type ='LCB',       # LCB acquisition

                                          acquisition_weight = 0.1)   # Exploration exploitation



x_best = np.exp(optimizer.X[np.argmin(optimizer.Y)])

print("Best parameters: learning_rate="+str(x_best[0])+",gamma="+str(x_best[1])+",max_depth="+str(x_best[2]))
