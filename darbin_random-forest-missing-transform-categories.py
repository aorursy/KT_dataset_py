# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

iowa_test_file_path = '../input/test.csv'



train_data = pd.read_csv(iowa_file_path)

test_data = pd.read_csv(iowa_test_file_path)

# Create target object and call it y

y = train_data.SalePrice

train_features = train_data.drop(['SalePrice'], axis = 1)

# Create X

#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#X = home_data[features]
# fill in missing numeric values

from sklearn.impute import SimpleImputer



#Impute

train_data_num = train_features.select_dtypes(exclude=['object'])

test_data_num = test_data.select_dtypes(exclude=['object'])

imputer = SimpleImputer()

train_num_cleaned = imputer.fit_transform(train_data_num)

test_num_cleaned = imputer.transform(test_data_num)



#columns rename after imputing

train_num_cleaned = pd.DataFrame(train_num_cleaned)

test_num_cleaned = pd.DataFrame(test_num_cleaned)



train_num_cleaned.columns = train_data_num.columns

test_num_cleaned.columns = test_data_num.columns
# string columns: transform to dummies

train_data_str = train_data.select_dtypes(include=['object'])

test_data_str = test_data.select_dtypes(include=['object'])

train_str_dummy = pd.get_dummies(train_data_str)

test_str_dummy = pd.get_dummies(test_data_str)

train_dummy, test_dummy = train_str_dummy.align(test_str_dummy, 

                                                join = 'left', 

                                                axis = 1)
print(train_num_cleaned.columns)

print(train_num_cleaned.index)

print(test_num_cleaned.columns)

print(test_num_cleaned.index)

print(train_dummy.columns)

print(train_dummy.index)

print(test_dummy.columns)

print(test_dummy.index)
# convert numpy to pandas DataFrame

train_num_cleaned = pd.DataFrame(train_num_cleaned)

test_num_cleaned = pd.DataFrame(test_num_cleaned)
# joining numeric and string data

train_all_clean = pd.concat([train_num_cleaned, train_dummy], axis = 1)

test_all_clean = pd.concat([test_num_cleaned, test_dummy], axis = 1)
# detect NaN in already cleaned test data 

# (there could be completely empty columns)

cols_with_missing = [col for col in test_all_clean.columns

                                if test_all_clean[col].isnull().any()]

for col in cols_with_missing:

    print(col, test_all_clean[col].describe())
# since there are empty columns in test we need to drop them in train and test

train_all_clean_no_nan = train_all_clean.drop(cols_with_missing, axis = 1)

test_all_clean_no_nan = test_all_clean.drop(cols_with_missing, axis = 1)
# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(train_all_clean_no_nan, y, random_state=1)



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

rf_model_on_full_data = RandomForestRegressor()



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(train_all_clean_no_nan, y)

test_X = test_all_clean_no_nan



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)
output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)