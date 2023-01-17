import numpy as np
import pandas as pd

from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error

%matplotlib inline

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
n = diabetes.data.shape[0]

data = diabetes.data
targets = diabetes.target
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split

rand=42
n_iter=48

train_X, test_X, train_y, test_y = train_test_split(data, targets, test_size=.2, 
                                                    shuffle=True, random_state=rand)

num_folds=2
kf = KFold(n_splits=num_folds, random_state=rand)
model = LGBMRegressor(random_state=rand)
%%time
score = -cross_val_score(model, train_X, train_y, cv=kf, 
                         scoring="neg_mean_squared_error", n_jobs=-1).mean()
print(score)
%%time
from sklearn.model_selection import GridSearchCV

param_grid={'learning_rate': np.logspace(-3,-1, 4),
            'max_depth':  np.linspace(5,12, 3, dtype=int),
            'n_estimators': np.linspace(800,1200, 4, dtype=int),
            'random_state': [rand]}

gs=GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', fit_params=None, 
                n_jobs=-1, cv=kf, verbose=False)

gs.fit(train_X, train_y)
gs_test_score=mean_squared_error(test_y, gs.predict(test_X))


print("Best MSE {:.3f} params {}".format(-gs.best_score_, gs.best_params_))
gs_results_df=pd.DataFrame(np.transpose([-gs.cv_results_['mean_test_score'],
                                         gs.cv_results_['param_learning_rate'].data,
                                         gs.cv_results_['param_max_depth'].data,
                                         gs.cv_results_['param_n_estimators'].data]),
                           columns=['score', 'learning_rate', 'max_depth', 'n_estimators'])
gs_results_df.plot(subplots=True,figsize=(12,8));
%%time
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_grid_rand={'learning_rate': np.logspace(-5, 0, 100),
                 'max_depth':  randint(2,20),
                 'n_estimators': randint(100,2000),
                 'random_state': [rand]}

rs=RandomizedSearchCV(model, param_grid_rand, n_iter = n_iter, 
                      scoring='neg_mean_squared_error', fit_params=None, 
                      n_jobs=-1, cv=kf, verbose=False, random_state=rand)

rs.fit(train_X, train_y)

rs_test_score=mean_squared_error(test_y, rs.predict(test_X))

print("Best MSE {:.3f} params {}".format(-rs.best_score_, rs.best_params_))
rs_results_df=pd.DataFrame(np.transpose([-rs.cv_results_['mean_test_score'],
                                         rs.cv_results_['param_learning_rate'].data,
                                         rs.cv_results_['param_max_depth'].data,
                                         rs.cv_results_['param_n_estimators'].data]),
                           columns=['score', 'learning_rate', 'max_depth', 'n_estimators'])
rs_results_df.plot(subplots=True,figsize=(12,8));
!pip install hyperopt
#!conda install -c conda-forge hyperopt
from hyperopt import fmin, tpe, hp, anneal, Trials
def gb_mse_cv(params, random_state=rand, cv=kf, X=train_X, y=train_y):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']), 
              'max_depth': int(params['max_depth']), 
             'learning_rate': params['learning_rate']}
    
    # we use this params to create a new LGBM Regressor
    model = LGBMRegressor(random_state=rand, **params)
    
    # and then conduct the cross validation with the same folds as before
    score = -cross_val_score(model, X, y, cv=cv, 
                             scoring="neg_mean_squared_error", n_jobs=-1).mean()

    return score
%%time

# possible values of parameters
space={'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
       'max_depth' : hp.quniform('max_depth', 2, 20, 1),
       'learning_rate': hp.loguniform('learning_rate', -5, 0)
      }

# trials will contain logging information
trials = Trials()

best=fmin(fn=gb_mse_cv, # function to optimize
          space=space, 
          algo=tpe.suggest, # optimization algorithm, hyperopt will select parameters automatically
          max_evals=n_iter, # max number of iterations
          trials=trials, # logging
          rstate=np.random.RandomState(rand) # fix random state for reproducibility
         )

# computing the score on the test set
model = LGBMRegressor(random_state=rand, n_estimators=int(best['n_estimators']),
                      max_depth=int(best['max_depth']),learning_rate=best['learning_rate'])
model.fit(train_X,train_y)
tpe_test_score=mean_squared_error(test_y, model.predict(test_X))

print("Best MSE {:.3f} params {}".format( gb_mse_cv(best), best))
tpe_results=np.array([[x['result']['loss'],
                      x['misc']['vals']['learning_rate'][0],
                      x['misc']['vals']['max_depth'][0],
                      x['misc']['vals']['n_estimators'][0]] for x in trials.trials])

tpe_results_df=pd.DataFrame(tpe_results,
                           columns=['score', 'learning_rate', 'max_depth', 'n_estimators'])
tpe_results_df.plot(subplots=True,figsize=(12,8));
%%time

# possible values of parameters
space={'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
       'max_depth' : hp.quniform('max_depth', 2, 20, 1),
       'learning_rate': hp.loguniform('learning_rate', -5, 0)
      }

# trials will contain logging information
trials = Trials()

best=fmin(fn=gb_mse_cv, # function to optimize
          space=space, 
          algo=anneal.suggest, # optimization algorithm
          max_evals=n_iter, # max number of iterations
          trials=trials, # logging
          rstate=np.random.RandomState(rand) # fix random state
         )

# computing the score on the test set
model = LGBMRegressor(random_state=rand, n_estimators=int(best['n_estimators']),
                      max_depth=int(best['max_depth']),learning_rate=best['learning_rate'])
model.fit(train_X, train_y)
sa_test_score=mean_squared_error(test_y, model.predict(test_X))

print("Best MSE {:.3f} params {}".format( gb_mse_cv(best), best))
sa_results=np.array([[x['result']['loss'],
                      x['misc']['vals']['learning_rate'][0],
                      x['misc']['vals']['max_depth'][0],
                      x['misc']['vals']['n_estimators'][0]] for x in trials.trials])

sa_results_df=pd.DataFrame(sa_results,
                           columns=['score', 'learning_rate', 'max_depth', 'n_estimators'])
sa_results_df.plot(subplots=True,figsize=(12,8));
scores_df=pd.DataFrame(index=range(n_iter))
scores_df['Grid Search']=gs_results_df['score'].cummin()
scores_df['Random Search']=rs_results_df['score'].cummin()
scores_df['TPE']=tpe_results_df['score'].cummin()
scores_df['Annealing']=sa_results_df['score'].cummin()

ax = scores_df.plot(figsize=(12,4))

ax.set_xlabel("number_of_iterations")
ax.set_ylabel("best_cumulative_score")
print('Test MSE scored:')
print("Grid Search {:.3f}".format(gs_test_score))
print("Random Search {:.3f}".format(rs_test_score))
print("TPE {:.3f}".format(tpe_test_score))
print("Annealing {:.3f}".format(sa_test_score))
# installing hpsklearn
!pip install hpsklearn
%%time
from hpsklearn import HyperoptEstimator, xgboost_regression

estim = HyperoptEstimator(regressor=xgboost_regression('my_gb'), 
                          max_evals=n_iter, trial_timeout=60, seed=rand)
estim.fit(train_X, train_y)

print("MSE:", mean_squared_error(test_y, estim.predict(test_X)))
est = HyperoptEstimator(max_evals=n_iter, trial_timeout=60, seed=rand)
est.fit(train_X, train_y)

print("MSE:", mean_squared_error(test_y, est.predict(test_X)))