# Code you have previously used to load data

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



cols = ['LotArea','OverallQual','OverallCond','YearBuilt', 'YearRemodAdd', 'TotalBsmtSF','1stFlrSF', '2ndFlrSF', 'GrLivArea','FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','GarageYrBlt','GarageArea','YrSold','SalePrice']

home_data = pd.read_csv(iowa_file_path,cols,delimiter=',')

home_data.fillna(value=0, inplace=True)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea','OverallQual','OverallCond','YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea','FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','GarageYrBlt','YrSold','GarageArea']

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

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

#test_data_path = '../input/test.csv'

#test_data = pd.read_csv(test_data_path)

#val_X = test_data[features]





sc=RobustScaler()

X=sc.fit_transform(X)

#val_X=sc.transform(val_X)



# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = XGBRegressor(random_state=1, max_depth=5, objective='reg:linear')



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)



#rf_model = RandomForestRegressor(random_state=1)

#rf_model.fit(train_X, train_y)

#rf_val_predictions = rf_model_on_full_data.predict(val_X)

#rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



#print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

#from sklearn.model_selection import GridSearchCV

#parameters = [{'n_estimators':np.random.random_integers(50, 600, 3), 

#                        'alpha' : [0.01, 0.1], 'eta' : np.logspace(-2, 0, 3, endpoint = False)}]



#grid_search= GridSearchCV(estimator=rf_model_on_full_data,

#                         param_grid=parameters,

#                         scoring='neg_mean_absolute_error',

#                         cv=10,

#                         n_jobs=-1)

#grid_search=grid_search.fit(train_X,train_y)
#mae=grid_search.best_score_

#mae
#grid_search.best_params_
# path to file you will use for predictions

test_data_path = '../input/home-data-for-ml-course/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



test_X=sc.transform(test_X)



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)