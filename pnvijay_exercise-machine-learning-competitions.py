import pandas as pd

import numpy as np

import tensorflow as tf

print(tf.__version__)

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)

home_data.describe()

home_data_predictors = home_data.drop(['SalePrice'],axis=1)

home_data_predictors = pd.get_dummies(home_data_predictors)

#home_data_numeric = home_data_predictors.select_dtypes(exclude=['object'])
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

y = home_data.SalePrice

# Create X

X = home_data_predictors

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]

scaler = MinMaxScaler()

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import MinMaxScaler



impute_train_X = train_X.copy()

impute_val_X = val_X.copy()



for col in cols_with_missing:

    impute_train_X[col+'_was_missing']= impute_train_X[col].isnull()

    impute_val_X[col+'_was_missing']= impute_val_X[col].isnull()

    

# impute_train_X.describe()

# impute_val_X.describe()



from sklearn.impute import SimpleImputer

Imputer = SimpleImputer()

train_X = Imputer.fit_transform(impute_train_X)

val_X = Imputer.transform(impute_val_X)

# train_X = scaler.fit_transform(train_X)

# val_X = scaler.transform(val_X)

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

from xgboost import XGBRegressor

xg_model = XGBRegressor(random_state=1)

xg_model.fit(train_X, train_y, verbose=False)

xg_val_predictions = xg_model.predict(val_X)

xg_val_mae = mean_absolute_error(xg_val_predictions, val_y)



print("Validation MAE for XGBoost Model: {:,.0f}".format(xg_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

import numpy as np

xg_model_on_full_data = XGBRegressor(n_estimators=200,learning_rate=0.5,random_state=1)

int_columns = [i for i in home_data_predictors.columns if home_data_predictors[i].dtype in [np.int64]]

X_new = home_data_predictors[int_columns]

xg_model_on_full_data.fit(X_new,y,verbose=False)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

# test_data.describe()
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

X_new = test_data[int_columns]

test_preds = xg_model_on_full_data.predict(X_new)





# The lines below shows you how to save your data in the format needed to score it in the competition

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})



output.to_csv('submission.csv', index=False)