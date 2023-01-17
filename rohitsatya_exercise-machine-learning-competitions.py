# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageArea']

#features = ['LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'MiscVal']

#X = home_data[features]



#categorical vars

s = (home_data.dtypes == 'object')

object_cols = list(s[s].index)





feature_cols_num = home_data._get_numeric_data().columns.drop(['Id', 'SalePrice'])

#X = home_data[feature_cols]



feature_cols = home_data.columns.drop(['Id', 'SalePrice'])

X = home_data[feature_cols]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



my_imputer = SimpleImputer(strategy='most_frequent')

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X[object_cols]))

imputed_X_train.columns = object_cols



imputed_X_val = pd.DataFrame(my_imputer.fit_transform(val_X[object_cols]))

imputed_X_val.columns = object_cols



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imputed_X_train))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(imputed_X_val))



# One-hot encoding removed index; put it back

OH_cols_train.index = imputed_X_train.index

OH_cols_valid.index = imputed_X_val.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = imputed_X_train.drop(object_cols, axis=1)

num_X_valid = imputed_X_val.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

#train_X = pd.concat([train_X[feature_cols_num], OH_cols_train], axis=1)

#val_X = pd.concat([val_X[feature_cols_num], OH_cols_valid], axis=1)





from sklearn.feature_selection import SelectKBest, f_classif



# Keep 5 features

selector = SelectKBest(f_classif, k=170)

#train_X.dropna()





my_imputer = SimpleImputer()



data_with_imputed_values = my_imputer.fit_transform(train_X[feature_cols_num.tolist()])



df = pd.DataFrame(data_with_imputed_values, columns = feature_cols_num.tolist())

df.columns = feature_cols_num.tolist()

train_X = pd.concat([df, OH_cols_train], axis=1)





X_new = selector.fit_transform(train_X, train_y)

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train_X.index, 

                                 columns=train_X.columns.tolist())

# Dropped columns have values of all 0s, so var is 0, drop them

selected_columns = selected_features.columns[selected_features.var() != 0]



# Get the valid dataset with the selected features.

features = selected_columns.tolist()



'''

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)



from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

val_X = my_imputer.fit_transform(val_X)

'''



data_with_imputed_values = my_imputer.fit_transform(val_X[feature_cols_num.tolist()])



df = pd.DataFrame(data_with_imputed_values, columns = feature_cols_num.tolist())

df.columns = feature_cols_num.tolist()

val_X = pd.concat([df, OH_cols_valid], axis=1)



train_X = train_X[features]

val_X = val_X[features]



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





from xgboost import XGBRegressor



xgb_model = XGBRegressor(n_estimator=1000)



from sklearn.model_selection import cross_val_score



xgb_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)

xgb_val_predictions = xgb_model.predict(val_X)

xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)

print("Validation MAE for XGB: {:,.0f}".format(xgb_val_mae))

# Multiply by -1 since sklearn calculates *negative* MAE

#scores = -1 * cross_val_score(xgb_model, X, y,

#                              cv=5,

#                              scoring='neg_mean_absolute_error')



#print("MAE scores:\n", scores)

#print("Average MAE score (across experiments):")

#print(scores.mean())
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

# To improve accuracy, create a new Random Forest model which you will train on all training data

#rf_model_on_full_data = RandomForestRegressor(random_state=1)



rf_model_on_full_data = XGBRegressor(n_estimator=100)



my_pipeline = Pipeline(steps=[('model', rf_model_on_full_data)])

#fit rf_model_on_full_data on all data from the training data

#rf_model_on_full_data.fit(X, y)



my_pipeline.fit(X, y)




# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]

#test_X.dropna()

#print(test_X.isnull().sum())



from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

data_with_imputed_values = my_imputer.fit_transform(test_X)



# make predictions which we will submit. 

#test_preds = rf_model_on_full_data.predict(data_with_imputed_values)

test_preds = my_pipeline.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
# Check your answer

step_1.check()

# step_1.solution()
import pandas as pd

test = pd.read_csv("../input/test.csv")
import pandas as pd

test = pd.read_csv("../input/test.csv")