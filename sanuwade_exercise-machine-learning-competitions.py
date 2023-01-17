# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

competition_file_path = '../input/train.csv'



home_data = pd.read_csv(competition_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ["LotArea","OverallQual","OverallCond","1stFlrSF","2ndFlrSF","GrLivArea","BedroomAbvGr","FullBath","GarageArea"]

X = home_data[features]

X = X.astype(float)



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)





# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

rf_mofd_pred = rf_model_on_full_data.predict(X)

rf_mofd_mae = mean_absolute_error(rf_mofd_pred, y)





print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

print("Validation MAE for Random Forest Model on Full Data: {:,.0f}".format(rf_mofd_mae))

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)





# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]

test_X = test_X.fillna(0)

print(test_X)





# make predictions which we will submit.



test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)