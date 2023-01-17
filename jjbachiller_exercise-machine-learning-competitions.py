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

#X = home_data[features]

X = home_data.drop(['SalePrice'], axis=1)

# Split into validation and training data

print(len(X.columns))

hot_encoded_X = pd.get_dummies(X)

print(len(hot_encoded_X.columns))

cols_with_missing = [col for col in hot_encoded_X.columns

                        if hot_encoded_X[col].isnull().any()]

reduced_X = hot_encoded_X.drop(cols_with_missing, axis=1)

print(len(reduced_X.columns))

#from sklearn.impute import SimpleImputer



#my_imputer = SimpleImputer()

#imputed_X = my_imputer.fit_transform(hot_encoded_X)



train_X, val_X, train_y, val_y = train_test_split(reduced_X, y, random_state=1)





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

rf_model_on_full_data = RandomForestRegressor(random_state = 1)

print(len(X.columns))

hot_encoded_X = pd.get_dummies(X)

print(len(hot_encoded_X.columns))

# make predictions which we will submit. 

#cols_with_missing = [col for col in hot_encoded_X.columns 

#                         if hot_encoded_X[col].isnull().any()]

#reduced_X = hot_encoded_X.drop(cols_with_missing, axis=1)

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_X = my_imputer.fit_transform(hot_encoded_X)

# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(imputed_X, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

print(len(test_data.columns))

#test_X = test_data[features]

# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features



#test_X = test_data.select_dtypes(exclude=['object'])

#hot-encoded data

test_X = pd.get_dummies(test_data)

# make predictions which we will submit. 

#cols_with_missing = [col for col in test_X.columns 

#                         if test_X[col].isnull().any()]

#reduced_test_X = test_X.drop(cols_with_missing, axis=1)

orig_X, final_test_X = hot_encoded_X.align(test_X, join='left', axis=1)

print(len(final_test_X.columns))

test_X_with_inputed_values = my_imputer.transform(final_test_X)

#print(len(final_test_X.columns))



#final_X, final_test = reduced_X.align(reduced_test_X, join='left', axis=1) 



#from sklearn.impute import SimpleImputer



#my_imputer = SimpleImputer()

#test_X_with_inputed_values = my_imputer.fit_transform(final_test_X)

#rf_model_on_full_data = RandomForestRegressor(random_state=1)

#rf_model_on_full_data.fit(test_X_with_inputed_values, train_y)

#test_preds = rf_model_on_full_data.predict(test_X_with_inputed_values)



#imputed_test_X = my_imputer.transform(test_X)

test_preds = rf_model_on_full_data.predict(test_X_with_inputed_values)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)