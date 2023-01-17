# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer









# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'





original_data = pd.read_csv(iowa_file_path)

home_data = original_data.copy()



cols_with_missing = [col for col in home_data.columns 

                                 if home_data[col].isnull().any()]

for col in cols_with_missing:

    home_data[col + '_was_missing'] = home_data[col].isnull()

   

# home_data = pd.read_csv(iowa_file_path)

# Imputation

# my_imputer = SimpleImputer()

# home_data = pd.DataFrame(my_imputer.fit_transform(original_data))

# home_data.columns = original_data.columns



# home_data = my_imputer.fit_transform(original_data)

home_data = original_data.drop(cols_with_missing, axis=1)



# missing_val_count_by_column = (home_data.isnull().sum())

# print(missing_val_count_by_column[missing_val_count_by_column > 0])



                                  

# Create target object and call it y



# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]

y = home_data.SalePrice



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=10)

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

rf_model = RandomForestRegressor(max_features=4,max_depth=20,random_state=10)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))





# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(max_depth=30,random_state=10)



# fit rf_model_on_full_data on all data from the 

rf_model_on_full_data.fit(X,y)

# rf_model_on_full_data_prediction=rf_model_on_full_data.predict(val_X)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)





# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows you how to save your data in the format needed to score it in the competition

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})



output.to_csv('submission2New.csv', index=False)
