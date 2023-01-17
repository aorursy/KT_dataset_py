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

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',

           'OverallCond','OverallQual', 'YearRemodAdd', 'PoolArea','GrLivArea','WoodDeckSF']



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
# Lets make it more useful by handling missing values and one hot encoding

from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer





# read test data file using pandas

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

idx = test_data.Id



# Drop houses where the target is missing

b4_td_shape = train_data.shape

print("Before dropping houses without target column", b4_td_shape)



# Drop columns with missing values

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice

train_data = train_data.drop(["SalePrice", "Id"], axis=1)

test_data = test_data.drop(["Id"], axis=1)

aft_td_shape = train_data.shape

print("After dropping houses without target column", aft_td_shape)

print("Test data", test_data.shape)

if b4_td_shape == aft_td_shape:

    print("Note: No missing Target columns in the data")

    

print(train_data.head())

print(test_data.head())



# select low cardinal columns and numerical columns

low_cardinality_cols = [cname for cname in train_data.columns if 

                                train_data[cname].nunique() < 10 and

                               train_data[cname].dtype == "object"]

numeric_cols = [cname for cname in train_data.columns if 

                                train_data[cname].dtype in ['int64', 'float64']]



# imputation done in numeric columns, 

# but we have to check whether any categorical columns have missing values

check = [col for col in train_data if train_data[col].isnull().any()]

for col in low_cardinality_cols:

    if col in check:

        low_cardinality_cols.remove(col)



# impute missing values in numerical columns

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_train = pd.DataFrame(my_imputer.fit_transform(train_data[numeric_cols]), columns=numeric_cols)

imputed_test = pd.DataFrame(my_imputer.transform(test_data[numeric_cols]), columns=numeric_cols)



# concatenate the imputed and low cardinal cols

train_predictors = pd.concat([imputed_train, pd.DataFrame(train_data[low_cardinality_cols], 

                                                        columns=low_cardinality_cols)], axis=1)

test_predictors = pd.concat([imputed_test, pd.DataFrame(test_data[low_cardinality_cols], 

                                                        columns=low_cardinality_cols)], axis=1)



one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)





final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='inner', 

                                                                    axis=1)

#Check any Nan

print([col for col in final_train if final_train[col].isnull().any()])

print([col for col in final_test if final_test[col].isnull().any()])



def get_mae(estimator, X, y):

    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention

    return -1 * cross_val_score(estimator, 

                                X, y, scoring = 'neg_mean_absolute_error').mean()



# Create Model to predict

rfModel = RandomForestRegressor(500, random_state=1)

# Calculating MAE

mae_one_hot_encoded = get_mae(rfModel, final_train, target)

print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))



# fit rf_model_on_full_data on all data from the training data to get the Trained Model

rfModel.fit(final_train, target)



# Predict The test data

# final_preds = rfModel.predict(final_test)

# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



#output = pd.DataFrame({'Id': idx,'SalePrice': final_preds})

#output.to_csv('submission.csv', index=False)
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05,silent=True, random_state=1)

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(final_train, target, verbose=False)

xgBoost_Score = get_mae(my_model, final_train, target)

print('Mean Abslute Error with XGBoost: ' + str(int(xgBoost_Score)))



# Predict The test data

final_preds = my_model.predict(final_test)

# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': idx,'SalePrice': final_preds})

output.to_csv('submission.csv', index=False)