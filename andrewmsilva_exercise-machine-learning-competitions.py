# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

from sklearn.impute import SimpleImputer







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)



features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',

            'KitchenAbvGr', 'BsmtFullBath', 'FullBath', 'GrLivArea', 'BsmtUnfSF', 'BsmtFinSF1', 'OverallQual',

            'OverallCond', 'GarageCars', 'Fireplaces', 'TotRmsAbvGrd']



target = home_data.SalePrice

predictors = home_data[features]

numeric_predictors = predictors.select_dtypes(exclude=['object'])



X_train, X_test, y_train, y_test = train_test_split(numeric_predictors, target, random_state=0)



my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(X_train)

imputed_X_test = my_imputer.transform(X_test)



model_options = {'n_estimators':118, 'max_leaf_nodes':125, 'random_state':0, 'min_samples_split': 4}



def score_dataset(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(**model_options)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds)



print("Mean Absolute Error from dropping columns with Missing Values:", score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

home_data.describe()



# 17145.716927138717
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(**model_options)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(numeric_predictors, target)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]

test_X = my_imputer.transform(test_X)



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)