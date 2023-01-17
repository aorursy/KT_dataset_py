import pandas as pd

import numpy as np



from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor



from sklearn.model_selection import KFold, cross_val_score
RANDOM_SEED=5



train = pd.read_csv('../input/idao2020/data/train.csv', index_col=0)

test =  pd.read_csv('../input/idao2020/data/Track 1/test.csv', index_col=0)
train.head(2)
test.head(2)
def prepare_features(df):

    '''minimal preprocessing'''

    date = pd.to_datetime(df.epoch)

    # year and month are the same accross the data

    df['day'] = date.dt.day

    df['weekday'] = date.dt.weekday

    df['hour'] = date.dt.hour

    df['minute'] = date.dt.minute

    df['second'] = date.dt.second

    

    return df.drop('epoch', axis=1)
train = prepare_features(train)

X = train[['x_sim', 'y_sim', 'z_sim',

           'Vx_sim', 'Vy_sim', 'Vz_sim',

           'sat_id', 'day', 'weekday', 'hour', 'minute','second']]

Y = train[['x', 'y', 'z',

           'Vx', 'Vy', 'Vz']]
from sklearn.model_selection import GridSearchCV
# Sattelite based cross-validation



rgn = RandomForestRegressor(n_estimators=10)

cv = list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X, Y['x'], groups=X['sat_id']))
params = {

    'max_depth': np.arange(3,8,2),

    'min_samples_split': np.arange(2,25,5),

    'min_samples_leaf': [5],

    'max_features': [0.3, 0.7],

    'random_state':[RANDOM_SEED],

}

gs = GridSearchCV(estimator=rgn,

                  param_grid=params,

                  scoring='neg_mean_squared_error',

                  cv=cv,

                  n_jobs=10,

                  verbose=5,

                  iid=False)
gs.fit(X, Y['x'])

# ~10 min
from sklearn.model_selection import RandomizedSearchCV

from scipy import stats
params = {

    'max_depth': stats.randint(2, 8),

    'min_samples_split': stats.randint(2, 25),

    'min_samples_leaf': [5],

    'max_features': stats.uniform(),

    'random_state':[RANDOM_SEED],

}





rs = RandomizedSearchCV(estimator=rgn,

                       param_distributions=params,

                       n_iter=30,

                       scoring='neg_mean_squared_error',

                       n_jobs=10,

                       cv=cv,

                       verbose=5,

                       random_state=RANDOM_SEED)
rs.fit(X, Y['x'])
from hyperopt import Trials, fmin, hp, tpe
rgn = RandomForestRegressor(n_estimators=10, min_samples_leaf=5, random_state=RANDOM_SEED)



def score(params):

    print(f"Training with params: {params}")

    rgn.set_params(**params)

    cv = list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(X, Y['x'], groups=X['sat_id']))

    neg_mse = cross_val_score(rgn, X, Y['x'], scoring='neg_mean_squared_error', cv=cv).mean()        

    return -neg_mse





def optimize(random_state=RANDOM_SEED, niter=2):

    param_space = {

    'max_depth': hp.choice('max_depth', np.arange(2, 8, dtype=int)),

    'min_samples_split': hp.choice('min_samples_split', np.arange(2, 25, dtype=int)),

    'max_features': hp.uniform('max_features',0, 1.),

    }

    trials = Trials()

    best = fmin(score, param_space, algo=tpe.suggest, 

                trials=trials, 

                max_evals=niter,

                rstate=np.random.RandomState(random_state)

               )

    return best, trials
# Use niter=2 for minimal example

best_hyperparams, trials = optimize(niter=30) 
# HyperOpt `fmin` returns indexes for `choice` defined parameters 



np.arange(2, 8, dtype=int)[5], np.arange(2, 25, dtype=int)[1]



best_hyperparams
# Checkout trials object



print(trials.results)

print(trials.best_trial)

print(trials.idxs_vals)
gs.best_params_
rs.best_params_
rgn = RandomForestRegressor(n_estimators=10, min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1)

rgn.set_params(**gs.best_params_)

cross_val_score(rgn, X, Y['x'], cv=cv, scoring='neg_mean_squared_error').mean()
rgn = RandomForestRegressor(n_estimators=10, min_samples_leaf=5, random_state=RANDOM_SEED, n_jobs=-1)

rgn.set_params(**rs.best_params_)

cross_val_score(rgn, X, Y['x'], cv=cv, scoring='neg_mean_squared_error').mean()
#{'max_depth': 5, 'max_features': 0.9336701952987806, 'min_samples_split': 1}



rgn = RandomForestRegressor(n_estimators=10, min_samples_leaf=5, max_features=0.9336701952987806,

                            min_samples_split = 3, max_depth=7,

                            random_state=RANDOM_SEED, n_jobs=-1)

cross_val_score(rgn, X, Y['x'], cv=cv, scoring='neg_mean_squared_error').mean()