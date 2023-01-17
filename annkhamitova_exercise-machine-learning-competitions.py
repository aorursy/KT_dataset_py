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

s = pd.get_dummies(home_data['MSZoning'])

s1 = pd.get_dummies(home_data['LotShape'])

s2 = pd.get_dummies(home_data['LotConfig'])

s3 = pd.get_dummies(home_data['HouseStyle'])

s4 = pd.get_dummies(home_data['Foundation'])

s5 = pd.get_dummies(home_data['Heating'])

s6 = pd.get_dummies(home_data['Electrical'])

s7 = pd.get_dummies(home_data['CentralAir'])

s8 = pd.get_dummies(home_data['Functional'])

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'KitchenAbvGr','Fireplaces','MSSubClass',

           'OverallQual','OverallCond','YearRemodAdd','GrLivArea','HalfBath']

X = home_data[features]

X1=pd.concat([X, s, s1, s2, s3, s4, s5, s6, s7, s8], axis=1) 

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X1, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



candidate_max_leaf_nodes=[100,150,200,250,300,350,400,450]

scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

best_tree_size = min(scores, key=scores.get)

print(best_tree_size)

# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

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

print(train_X)
s=pd.get_dummies(home_data['MSZoning'])

print(s)

X1=pd.concat([X, s,s1], axis=1)

print(X1)

home_data.describe()
home_data.head()
help(RandomForestRegressor)
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(60,criterion='mae',random_state=1,warm_start=True)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)

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
