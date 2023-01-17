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

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

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

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



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



#output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': test_preds})

#output.to_csv('submission.csv', index=False)
# see list of all features

print(f"Features used in initial model: {features}\n")

print(home_data.columns.tolist())
home_data.head()
home_data.describe()
test_data_columns = test_data.columns.tolist()

print(f"Columns in test data: {test_data_columns}")
# create new features for trial 1: 

t1_features = []



# add old features

for f in features:

    if f in test_data_columns:

        t1_features.append(f)

    

# new features

for new_f in ['OverallQual', 'OverallCond', 'GrLivArea', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd']:

    if new_f not in t1_features and new_f in test_data_columns:

        t1_features.append(new)

        

print(f"Trial 1 features: {t1_features}")
# trial 1 data

t1_X = home_data[t1_features]

t1_X.head()
t1_X.describe()
t1_X.info()
# creat new model

t1_rf_model = RandomForestRegressor(random_state=1)



# fit new model

t1_rf_model.fit(t1_X, y)
# trial 1 test data

t1_test_X = test_data[t1_features]

t1_test_X.describe()
t1_test_X.info()
# make predictions

t1_preds = t1_rf_model.predict(t1_test_X)



#output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': t1_preds})

#output.to_csv('submission2.csv', index=False)
# creat a function to fit the model and calculate the MAE

def rf_calc_mea(max_leaf_nodes, train_X, val_X, train_y, val_y):

    # create random forest (rf) model

    rf_model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)

    

    # fit the model

    rf_model.fit(train_X, train_y)

    

    # predict

    pred_y = rf_model.predict(val_X)

    

    # calculate MAE

    return mean_absolute_error(val_y, pred_y)

    

    
# list of possible number of leaf nodes

possible_leaf_nodes = range(100,401,10)



# split data

train_X, val_X, train_y, val_y = train_test_split(t1_X, y, random_state=1)



# list to save MAEs calculated

all_mae = []



# calculate MAEs

for leaf_nodes in possible_leaf_nodes:

    all_mae.append(rf_calc_mea(max_leaf_nodes=leaf_nodes, train_X=train_X, val_X=val_X, train_y=train_y, val_y=val_y))
print(all_mae)

best_tree_index = all_mae.index(min(all_mae))

best_tree_size = possible_leaf_nodes[best_tree_index]

print(f"\nThe best tree size is {best_tree_size}.")
# fit t1_X assigning the max_leaf_nodes option

# creat new model

t2_rf_model = RandomForestRegressor(max_leaf_nodes=best_tree_size, random_state=1)



# fit new model

t2_rf_model.fit(t1_X, y)
# make predictions

t2_preds = t2_rf_model.predict(t1_test_X)



#output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': t2_preds})

#output.to_csv('submission3.csv', index=False)
home_data.head()
# add 2 new columns to the original data

home_data['YearsOld'] = 2019 - home_data['YearBuilt']

home_data['YearsRemod'] = 2019 - home_data['YearRemodAdd']

home_data.head(10)
# add the same columns that are added to home_data to the test_data

test_data['YearsOld'] = 2019 - test_data['YearBuilt']

test_data['YearsRemod'] = 2019 - test_data['YearRemodAdd']

test_data.head(10)
test_data_columns = test_data.columns.tolist()

print(test_data_columns)
# features to add

to_add_features = ['MSSubClass', 'LotFrontage', 'YearsOld', 'YearsRemod', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF',

                  'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF',

                  'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'GarageCars', 'GarageArea']

t3_features = []



for f in to_add_features:

    if f not in t1_features and f in test_data_columns:

        t3_features.append(f)



# features to remove

remove_features = ['YearBuilt']

for r_f in remove_features:

    t3_features.pop(t3_features.index(r_f))



print(t3_features)
t3_data = home_data[t3_features]

t3_data.info()
# t3 colums are the features of t3 and the prediction variable 'SalePrice'

t3_colums = [f for f in t3_features]

t3_colums.append('SalePrice')

print(t3_colums)
t3_data = home_data[t3_colums].dropna(axis=0)

t3_data.info()
# t3 training and prediction data

t3_X = t3_data[t3_features]

t3_y = t3_data['SalePrice']
t3_test_colums = [f for f in t3_data if f != 'SalePrice']

t3_test_colums.append('Id')



# remove nulls from test_data too

t3_test_data = test_data[t3_test_colums].dropna(axis=0)

t3_test_data.info()
t3_test_X = t3_test_data[t3_features]
# creat t3 model

t3_rf_model = RandomForestRegressor(random_state=1)



# fit the model

t3_rf_model.fit(t3_X, t3_y)
t3_preds = t3_rf_model.predict(t3_test_X)



output = pd.DataFrame({'Id': t3_test_data.Id,

                       'SalePrice': t3_preds})

output.to_csv('submission4.csv', index=False)