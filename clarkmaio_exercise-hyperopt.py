import pandas as pd

import numpy as np

import time



from hyperopt import hp, tpe, Trials, fmin

max_evals = 100 # Global variable that define the number of iteration of optimisation algorithm
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split



data = load_boston()

X = pd.DataFrame(data['data'], columns = data['feature_names'])

y = pd.DataFrame(data['target'], columns = ['Price'])

X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, train_size = .8)

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, train_size = .6)





print('Features space: \n {}'.format(X.describe()))

print('\n\n')

print('Target space: \n {}'.format(y.describe()))
# Function to optimise

def objective(x):

    return x**2



# Define domain

space = hp.uniform('x', -1, 3)



# Define algorithm

algo = tpe.suggest



# Define trace

trials = Trials()



# All toghether now

t = time.time()

tpe_best = fmin(fn=objective, space=space, 

                algo=algo, trials=trials, 

                max_evals=max_evals)

delta_t = time.time()-t



# Result

print(tpe_best)

print('Optimisation completed in {} seconds'.format(np.round(delta_t,1)))
# We can have a look at the optimisation process using trials variable

tpe_results = pd.DataFrame({'loss': [x['loss'] for x in trials.results], 

                            'iteration': trials.idxs_vals[0]['x'],

                            'x': trials.idxs_vals[1]['x']})

tpe_results.head(10)
tpe_results.hist('x')
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(random_state=42)

mdl.fit(X_train, y_train.values.reshape(-1))

y_pred = mdl.predict(X_test)



print(mdl.base_estimator)



mae_default = np.mean(np.abs(y_pred - y_test.values))

print('MAE using default hyperparams: {}'.format(mae_default))
# Define function to minimize

def objective_sklearn(params,

                      X_train = X_train, y_train = y_train.values.reshape(-1),

                      X_test = X_val, y_test = y_val.values.reshape(-1)):

    

    # Make sure params are in the correct format

    params['n_estimators'] = int(params['n_estimators']) 

    params['max_depth'] = int(params['max_depth'])

    params['min_samples_split'] = int(params['min_samples_split']) 

              

    # Define the model using params

    mdl = RandomForestRegressor(random_state = 42, **params)

    mdl.fit(X_train, y_train)

    

    y_pred = mdl.predict(X_test)

    mae = np.mean(np.abs(y_pred - y_test))

    return mae



# Define domain

space_sklearn = {'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),

                 'max_depth' : hp.quniform('max_depth', 2, 20, 1),

                 'min_samples_split': hp.quniform('min_samples_split', 2, 100, 1)}



# Define algorithm

algo_sklearn = tpe.suggest



# Define trace

trials_sklearn = Trials()

t = time.time()

tpe_best_sklearn = fmin(fn=objective_sklearn, space=space_sklearn, 

                algo=algo_sklearn, trials=trials_sklearn,

               max_evals=max_evals)

delta_t = time.time()-t

print('Optimisation completed in {} seconds'.format(np.round(delta_t,0)))
print('Optimal params:')

print('n_estimators: {}'.format(tpe_best_sklearn['n_estimators']))

print('max_depth: {}'.format(tpe_best_sklearn['max_depth']))

print('min_samples_split: {}'.format(tpe_best_sklearn['min_samples_split']))
mdl = RandomForestRegressor(random_state=42,

                            max_depth=int(tpe_best_sklearn['max_depth']),

                            n_estimators=int(tpe_best_sklearn['n_estimators']),

                            min_samples_split=int(tpe_best_sklearn['min_samples_split']))

mdl.fit(X_train, y_train.values.reshape(-1))

y_pred = mdl.predict(X_test)



print(mdl.base_estimator)



mae_opt = np.mean(np.abs(y_pred - y_test.values))

print('MAE using optimal hyperparams: {}'.format(mae_opt))
print("MAE passes from {} to {}... that's a {}% improvment :)".format(mae_default, mae_opt, np.round((1-mae_default/mae_opt)*100, 2)))
import xgboost as xgb
mdl_xgb = xgb.XGBRegressor(random_state=42, objective ='reg:squarederror')

mdl_xgb.fit(X_train, y_train.values.reshape(-1))

y_pred = mdl_xgb.predict(X_test)



print(mdl_xgb)



mae_default = np.mean(np.abs(y_pred - y_test.values))

print('MAE using default hyperparams: {}'.format(mae_default))
# Define function to minimize

def objective_xgboost(params,

                      X_train = X_train, y_train = y_train.values.reshape(-1),

                      X_test = X_val, y_test = y_val.values.reshape(-1)):

    

    # Make sure params are in the correct format

    params['n_estimators'] = int(params['n_estimators']) 

    params['max_depth'] = int(params['max_depth'])

    params['learning_rate'] = float(params['learning_rate']) 

              

    # Define the model using params

    mdl = xgb.XGBRegressor(random_state = 42, objective ='reg:squarederror',

                           **params)

    mdl.fit(X_train, y_train)

    

    y_pred = mdl.predict(X_test)

    mae = np.mean(np.abs(y_pred - y_test))

    return mae



# Define domain

space_xgboost = {'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),

                 'max_depth' : hp.quniform('max_depth', 2, 20, 1),

                 'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.5))}



# Define algorithm

algo_xgboost = tpe.suggest



# Define trace

trials_xgboost = Trials()

t = time.time()

tpe_best_xgboost = fmin(fn=objective_xgboost, space=space_xgboost, 

                algo=algo_xgboost, trials=trials_xgboost,

               max_evals=max_evals)



delta_t = time.time()-t

print('Optimisation completed in {} seconds'.format(np.round(delta_t,0)))
print('Optimal params:')

print('n_estimators: {}'.format(tpe_best_xgboost['n_estimators']))

print('max_depth: {}'.format(tpe_best_xgboost['max_depth']))

print('learning_rate: {}'.format(tpe_best_xgboost['learning_rate']))
mdl = xgb.XGBRegressor(random_state=42,

                       objective ='reg:squarederror',

                            max_depth=int(tpe_best_xgboost['max_depth']),

                            n_estimators=int(tpe_best_xgboost['n_estimators']),

                            learning_rate=float(tpe_best_xgboost['learning_rate']))

mdl.fit(X_train, y_train.values.reshape(-1))

y_pred = mdl.predict(X_test)



mae_opt = np.mean(np.abs(y_pred - y_test.values))

print('MAE using optimal hyperparams: {}'.format(mae_opt))
print("MAE passes from {} to {}... that's a {}% improvment :)".format(mae_default, mae_opt, np.round((1-mae_default/mae_opt)*100, 2)))
from keras.models import Sequential

from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
# Build the model

def build_keras_mdl():

    mdl = Sequential()

    mdl.add(Dense(10, input_shape = (X.shape[1],), activation = 'relu'))

    mdl.add(Dense(5, activation = 'relu'))

    mdl.add(Dense(1, activation = 'relu'))



    mdl.compile(optimizer = 'rmsprop', loss = 'mae', metrics = ['mae'])

    return mdl



mdl_keras = build_keras_mdl()

print(mdl_keras.summary())
# Wwe need to scale the feature space

scaler = StandardScaler()

X_train_scale = scaler.fit_transform(X_train)

X_train_scale = pd.DataFrame(data =X_train_scale, columns = X_train.columns )



X_val_scale = scaler.transform(X_val)

X_val_scale = pd.DataFrame(data =X_val_scale, columns = X_val.columns )



X_test_scale = scaler.transform(X_test)

X_test_scale = pd.DataFrame(data =X_test_scale, columns = X_test.columns )
mdl_keras.fit(X_train_scale, y_train, batch_size = 32, epochs = 1)

y_pred = mdl_keras.predict(X_test_scale)



print(mdl_keras)



mae_default = np.mean(np.abs(y_pred - y_test.values))

print('MAE using default hyperparams: {}'.format(mae_default))
# Define function to minimize

def objective_keras(params,

                      X_train = X_train_scale, y_train = y_train.values.reshape(-1),

                      X_test = X_val_scale, y_test = y_val.values.reshape(-1)):

    

    # Make sure params are in the correct format

    params['batch_size'] = int(params['batch_size']) 

    params['epochs'] = int(params['epochs'])

              

    # Define the model using params

    mdl = build_keras_mdl()

    mdl.fit(X_train, y_train, verbose = 0, **params)

    

    y_pred = mdl.predict(X_test)

    mae = np.mean(np.abs(y_pred - y_test))

    return mae



# Define domain

space_keras = {'batch_size': hp.quniform('batch_size', 8, 128, 8),

                 'epochs' : hp.quniform('epochs', 1, 100, 1)}



# Define algorithm

algo_keras = tpe.suggest



# Define trace

trials_keras = Trials()

t = time.time()

tpe_best_keras = fmin(fn=objective_keras, space=space_keras, 

                algo=algo_keras, trials=trials_keras,

               max_evals=max_evals)



delta_t = time.time()-t

print('Optimisation completed in {} seconds'.format(np.round(delta_t,0)))
print('Optimal params:')

print('batch_size: {}'.format(tpe_best_keras['batch_size']))

print('epochs: {}'.format(tpe_best_keras['epochs']))
mdl = build_keras_mdl()

mdl.fit(X_train_scale, y_train.values.reshape(-1), batch_size = int(tpe_best_keras['batch_size']), epochs = int(tpe_best_keras['epochs']), verbose = 0)

y_pred = mdl.predict(X_test_scale)



mae_opt = np.mean(np.abs(y_pred - y_test.values))

print('MAE using optimal hyperparams: {}'.format(mae_opt))
print("MAE passes from {} to {}... that's a {}% improvment :)".format(mae_default, mae_opt, np.round((1-mae_default/mae_opt)*100, 2)))