# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','MSSubClass','OverallQual','OverallCond','YearRemodAdd','YrSold'

,'MoSold','WoodDeckSF','OpenPorchSF','3SsnPorch','ScreenPorch']

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

test_X, val_X, test_y, val_y = train_test_split(val_X, val_y, random_state=1)
home_data.describe()#
import matplotlib.pyplot as plt 

%matplotlib inline

#choose the best n_estimator parameter

n_estimators = range(10,300)

Mae = []

for n in n_estimators :

    rf_model = RandomForestRegressor(n_estimators= n ,random_state=1)

    rf_model.fit(train_X, train_y)

    rf_val_predictions = rf_model.predict(val_X)

    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    Mae.append(rf_val_mae)

    

plt.plot(n_estimators,Mae) 



rf_model = RandomForestRegressor(n_estimators= 280,random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

#best max_depth

Mae_f = []

f_range = range(10,1000,10)

for i in f_range :

    rf_model = RandomForestRegressor(max_depth =i,n_estimators=280, random_state=1)

    rf_model.fit(train_X, train_y)

    rf_val_predictions = rf_model.predict(val_X)

    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    Mae_f.append(rf_val_mae)

print("Validation MAE for Random Forest Model: {:,.0f}".format(f_range[Mae_f.index(min(Mae_f))]))
#best max_leaf_nodes

Mae_f = []

f_range = range(10,1000,10)

for i in f_range :

    rf_model = RandomForestRegressor(max_leaf_nodes=i,n_estimators=280, random_state=1)

    rf_model.fit(train_X, train_y)

    rf_val_predictions = rf_model.predict(val_X)

    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    Mae_f.append(rf_val_mae)

print("Validation MAE for Random Forest Model: {:,.0f}".format(f_range[Mae_f.index(min(Mae_f))]))
#best maxfeature

Mae_f = []

f_range = range(1,len(features))

for i in f_range :

    rf_model = RandomForestRegressor(max_features =i,n_estimators= 280,max_depth=40,random_state=1)

    rf_model.fit(train_X, train_y)

    rf_val_predictions = rf_model.predict(val_X)

    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    Mae_f.append(rf_val_mae)

print("Validation MAE for Random Forest Model: {:,.0f}".format(f_range[Mae_f.index(min(Mae_f))]))
rf_model = RandomForestRegressor(n_estimators= 280,max_depth=40,max_features=11,max_leaf_nodes=350,random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(test_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, test_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
rf_model_on_full_data = RandomForestRegressor(n_estimators= 280,max_depth = 40,max_features=15,max_leaf_nodes=350,random_state=1)

rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



test_data = test_data.dropna(axis=1)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features



test_X = test_data[features]

#test_y = test_data.SalePrice

# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)

#rf_val_mae = mean_absolute_error(test_preds, test_y)

#print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)