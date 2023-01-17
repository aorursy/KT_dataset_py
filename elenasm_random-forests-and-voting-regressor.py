#import packages I'll need



import pandas as pd

import numpy as np



from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



from sklearn.metrics import mean_squared_error

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#read database

df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
#standardize the data, even if for Random Forests is not necessary I'll be using Linear Regression later



scaler = preprocessing.StandardScaler()



#train/test split

train_db, test_db  = train_test_split(df, test_size = 0.2, random_state = 10)



#isolate dv

train_label = train_db['Chance of Admit '].copy()

test_label = test_db['Chance of Admit '].copy()



train_db = train_db.drop(['Chance of Admit '], axis = 1)

test_db = test_db.drop(['Chance of Admit '], axis = 1)



#standardization

train_db_scaled = scaler.fit_transform(train_db)

test_db_scaled = scaler.transform(test_db)
#test Random Forests without any tuning



forest_reg = RandomForestRegressor()



forest_reg.fit(train_db_scaled, train_label)

predictions_forest = forest_reg.predict(test_db_scaled)

mse_forest = mean_squared_error(predictions_forest, test_label)

rmse_forest = np.sqrt(mse_forest)

rmse_forest
# random forests with hyperparameters



# let's see what hyperparameters we have now

from pprint import pprint

pprint(forest_reg.get_params())
#  Random Hyperparameter Grid Search





from sklearn.model_selection import RandomizedSearchCV



n_estimators = [int(x) for x in np.linspace(start = 5, stop = 800, num = 30)]



max_features = ['auto', 'sqrt']



max_depth = [int(x) for x in np.linspace(5, 2000, num = 30)]

max_depth.append(None)



min_samples_split = [2, 4, 6, 10]



min_samples_leaf = [1,2, 4, 6, 10]



bootstrap = [True, False]



# Create the random grid I'll use at the next step

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)
#test 100 random parameter combinations, with 3 fold cross validation



forest_reg2 = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = forest_reg2, param_distributions = random_grid, n_iter = 100, cv = 3, 

                               verbose=2, random_state=10, n_jobs = -1)



rf_random.fit(train_db_scaled, train_label)
#check best parameters



rf_random.best_params_
#define a parameter grid close to what seemed to be the best parameters from

#the random search and test each one with Grid Search



from sklearn.model_selection import GridSearchCV



param_grid = {

    'bootstrap': [True],

    'max_depth': [500,800,1000,1200, 1400, 1500],

    'max_features': ['sqrt'],

    'min_samples_leaf': [1,2,3,4],

    'min_samples_split': [3,4,5,6],

    'n_estimators': [600,700,750,800]

}





rf = RandomForestRegressor()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search.fit(train_db_scaled, train_label)

grid_search.best_params_
#let's see the best choice

best_grid = grid_search.best_estimator_

best_grid
#now let's check the new rmse

predictions_grid = best_grid.predict(test_db_scaled)

mse_forest_grid = mean_squared_error(predictions_grid, test_label)

rmse_forest_grid = np.sqrt(mse_forest_grid)

rmse_forest_grid
#test regression



lin_reg = LinearRegression()



lin_reg.fit(train_db_scaled, train_label)

predictions_lin = lin_reg.predict(test_db_scaled)

mse_lin = mean_squared_error(predictions_lin, test_label)

rmse_lin = np.sqrt(mse_lin)

rmse_lin
# test Voting Regressor



from sklearn.ensemble import VotingRegressor



voting_reg = VotingRegressor(estimators=[('lr', lin_reg), ('fr', forest_reg), ('bg', best_grid)])

voting_reg.fit(train_db_scaled, train_label)



predictions_voting_reg = voting_reg.predict(test_db_scaled)

mse_v = mean_squared_error(predictions_voting_reg, test_label)

rmse_v = np.sqrt(mse_v)

rmse_v