# Code you have previously used to load data

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

print(home_data.describe())



# Create target object and call it y

y = home_data.SalePrice



# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'OverallCond', 'YearRemodAdd', 'OverallQual', 'PoolArea']

X = home_data[features]





# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)



# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# parameter grid sample

n_estimators = [int(n) for n in np.linspace(start=200, stop=2000, num=10)]

max_features = ['auto', 'sqrt']

max_depth = [int(n) for n in np.linspace(10, 110, num=11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4, 8]

bootstrap = [True, False]



# create the random feature grid

exp_random_grid = {'n_estimators': n_estimators,

                   'max_features': max_features,

                   'max_depth': max_depth,

                   'min_samples_split': min_samples_split,

                   'min_samples_leaf': min_samples_leaf,

                   'bootstrap': bootstrap}



# create our new Random Forest (experiment) (base to tune)

optimized_rf = RandomForestRegressor()



# Run Random Search of our specified parameters, using 3 fold cross validation

# Search across 100 combinations and use all cores available

#commented out the below lines after determining the best parameters because of the long running time.

#rf_random_search = RandomizedSearchCV(estimator=optimized_rf, param_distributions=exp_random_grid, n_iter=100, cv=3, random_state=42, n_jobs=-1)

#rf_random_search.fit(train_X, train_y)



#print our optimal params

#print(rf_random_search.best_params_)



best_params_rf = RandomForestRegressor(n_estimators=2000, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=60, bootstrap=False, random_state=42)

best_params_rf.fit(train_X, train_y)

best_params_predictions = best_params_rf.predict(val_X)

best_params_rf_mae = mean_absolute_error(best_params_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(best_params_rf_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data (old model)

#rf_model_on_full_data = RandomForestRegressor(n_estimators=100, random_state=1)



rf_model_optimal_params_full_data = RandomForestRegressor(n_estimators=2000, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=60, bootstrap=False, random_state=1)

# fit rf_model_on_full_data on all data from the training data

rf_model_optimal_params_full_data.fit(X, y)
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
# Check your answer

step_1.check()

# step_1.solution()