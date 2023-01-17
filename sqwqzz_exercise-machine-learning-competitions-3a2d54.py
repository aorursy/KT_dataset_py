# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

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

rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.

test_features = ['OverallQual', 'OverallCond','LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'KitchenAbvGr', 'GarageArea']

test_X = test_data[test_features]



more_features =  ['OverallQual', 'OverallCond','LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'KitchenAbvGr', 'GarageArea']

MX = home_data[more_features]

X_train, X_val, y_train, y_val = train_test_split(MX, y, random_state=1)



test_data.describe()

one_hot_encoded_training_predictors = pd.get_dummies(X_train)
def get_mae(X, y):

    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention

    return -1 * cross_val_score(RandomForestRegressor(50), 

                                X, y, 

                                scoring = 'neg_mean_absolute_error').mean()



predictors_without_categoricals = X_train.select_dtypes(exclude=['object'])



mae_without_categoricals = get_mae(predictors_without_categoricals, y_train)



mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, y_train)



print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))

print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
imputed_X_train = X_train.copy()

imputed_X_val = X_val.copy()

imputed_X_test = test_X.copy()



cols_with_missing = (col for col in X_train.columns 

                                 if X_train[col].isnull().any())

for col in cols_with_missing:

    imputed_X_train[col + '_was_missing'] = imputed_X_train[col].isnull()

    imputed_X_val[col + '_was_missing'] = imputed_X_val[col].isnull()

    imputed_X_test[col + '_was_missing'] = imputed_X_test[col].isnull()

from sklearn.impute import SimpleImputer

# Imputation

my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(imputed_X_train)

imputed_X_val = my_imputer.transform(imputed_X_val)

imputed_X_test = my_imputer.transform(imputed_X_test)
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_imputed_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_imputed_data.fit(imputed_X_train, y_train)

#rf_val_imputed_predictions = rf_model_on_imputed_data.predict(X_test)

#rf_val_imputed_mae = mean_absolute_error(rf_val_imputed_predictions, y_test)



#print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_imputed_mae))

from xgboost import XGBRegressor

XGmy_model = XGBRegressor(n_estimators=1000)

XGmy_model.fit(imputed_X_train, y_train, early_stopping_rounds=12, 

             eval_set=[(imputed_X_val, y_val)], verbose=False)



# make predictions

predictions = XGmy_model.predict(imputed_X_val)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_val)))


# make predictions which we will submit. 

#test_preds = rf_model_on_imputed_data.predict(imputed_X_test)

test_preds = XGmy_model.predict(imputed_X_test)

# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)