# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *





def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



# Path of the file to read. We changed the directory structure to simplify submitting to a competition



iowa_file_path = '../input/train_day_7.csv'

home_data = pd.read_csv(iowa_file_path)

home_data = home_data.dropna(axis=0)



# Create target object and call it y

y = home_data.SalePrice

 

# Create X



features = list(home_data.columns.values)

features = features[0:-1]

print(features)



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

#print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))





# Using best value for max_leaf_nodes

##original data had best value of 100, appears to be 500 for the RandomForestRegressor model 

##iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

rf_model=RandomForestRegressor(max_leaf_nodes=500, random_state=1)

rf_model.fit(train_X, train_y)

val_predictions = rf_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



#compare MAE with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))





# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



##print(rf_model.fit(train_X, train_y))





### Original results using features  ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

### Validation MAE when not specifying max_leaf_nodes: 29,653

### Validation MAE for best value of max_leaf_nodes: 27,283

### Validation MAE for Random Forest Model: 22,762





# Using the following features got the results below:  ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# Validation MAE when not specifying max_leaf_nodes: 25,034

# Validation MAE for best value of max_leaf_nodes: 23,120

# Validation MAE for Random Forest Model: 21,082







# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = 25247



# fit rf_model_on_full_data on all data from the training data

21082

# path to file you will use for predictions

test_data_path = '../input/train_day_7.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = list(home_data.columns.values)

features = test_X[0:-1]



# make predictions which we will submit. 

test_preds = rf_model.predict(val_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submissrf_model.predict(val_X)ion.csv', index=False)