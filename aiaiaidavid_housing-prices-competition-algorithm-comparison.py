import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read the train.csv file (training / validation data) and test.csv (testing / prediction data)

X = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')
# Check up nulls (training set)

X.isnull().sum()[X.isnull().any()]
# Check up nulls (testing set)

X_test.isnull().sum()[X_test.isnull().any()]
cols_to_drop = ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']

X.drop(cols_to_drop, axis=1, inplace=True)

X_test.drop(cols_to_drop, axis=1, inplace=True)
# Separate target variable from predicting variables

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)
# Display stats for categorical columns: training set

X.describe(include=['object'], exclude=['int64', 'float64'])
# Display stats for categorical columns: testing set

X_test.describe(include=['object'], exclude=['int64', 'float64'])
# Object-type columns with null values: training set

X_obj_cols_with_nulls = [[col, X[col].isnull().sum()] for col in X.columns if X[col].dtype == "object" and X[col].isnull().sum()>0]

print(len(X_obj_cols_with_nulls))

X_obj_cols_with_nulls
# Object-type columns with null values: testing set

X_test_obj_cols_with_nulls = [[col, X_test[col].isnull().sum()] for col in X_test.columns if X_test[col].dtype == "object" and X_test[col].isnull().sum()>0]

print(len(X_test_obj_cols_with_nulls))

X_test_obj_cols_with_nulls
X_cols_with_nulls = [col for col in X.columns if X[col].dtype == "object" and X[col].isnull().sum()>0]

for col in X_cols_with_nulls:

    X[col].fillna(value='Unknown', inplace=True)
X_test_cols_with_nulls = [col for col in X_test.columns if X_test[col].dtype == "object" and X_test[col].isnull().sum()>0]

for col in X_test_cols_with_nulls:

    X_test[col].fillna(value='Unknown', inplace=True)
from sklearn.preprocessing import LabelEncoder

object_cols = [col for col in X.columns if X[col].dtype == "object"]

# Apply label encoder to categorical columns

label_encoder = LabelEncoder()

for col in object_cols:

    val1 = X[col].unique()

    val2 = X_test[col].unique()

    values = list(set().union(val1, val2))

    label_encoder.fit(values)

    X[col] = label_encoder.transform(X[col])

    X_test[col] = label_encoder.transform(X_test[col])  
# Number of missing values in each column of training data

missing_val_count_by_column_1 = (X.isnull().sum())

print(missing_val_count_by_column_1[missing_val_count_by_column_1 > 0])
# Number of missing values in each column of test data

missing_val_count_by_column_2 = (X_test.isnull().sum())

print(missing_val_count_by_column_2[missing_val_count_by_column_2 > 0])
# Numerical columns missing in training set

num_cols_with_nulls = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']



X[num_cols_with_nulls].describe()
# Fill nulls independently

X['LotFrontage'].fillna(value=X['LotFrontage'].quantile(0.75), inplace=True)

X_test['LotFrontage'].fillna(value=X_test['LotFrontage'].quantile(0.75), inplace=True)

X['MasVnrArea'].fillna(value=X['MasVnrArea'].mean(), inplace=True)

X_test['MasVnrArea'].fillna(value=X_test['MasVnrArea'].mean(), inplace=True)

X['GarageYrBlt'].fillna(value=X['YearBuilt'], inplace=True)

X_test['GarageYrBlt'].fillna(value=X_test['YearBuilt'], inplace=True)
# Numerical columns missing in testing set

cols_with_missing_2 = [col for col in X_test.columns if X_test[col].isnull().any()]



X_test[cols_with_missing_2].describe()
# Fill nulls with column mean

for col in cols_with_missing_2:

    X_test[col].fillna(value=X_test[col].mean(), inplace=True)
from sklearn.model_selection import train_test_split



# Split data into train and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
%%time

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Define and fit model

model_1 = RandomForestRegressor(n_estimators=4000, random_state=0, n_jobs=4)

model_1.fit(X_train, y_train)



# Get validation predictions and MAE

preds_valid_1 = model_1.predict(X_valid)

print("MAE_1 (Random Forest Regressor): " + str(mean_absolute_error(y_valid, preds_valid_1)))
# Define and fit model with full train data

# X_train = X.copy()

# y_train = y.copy()

# model_1 = RandomForestRegressor(n_estimators=500, random_state=0)

# model_1.fit(X_train, y_train)



# # Make test predictions

# preds_test_1 = model_1.predict(X_test)
%%time

from xgboost import XGBRegressor



# Define and fit the model

model_2 = XGBRegressor(n_estimators=4000, learning_rate=0.015, n_jobs=4)

model_2.fit(X_train, y_train)



# Get validation predictions and MAE

preds_valid_2 = model_2.predict(X_valid)

print("MAE_2 (XG Boost Regressor): " + str(mean_absolute_error(y_valid, preds_valid_2)))
# # Define and fit model with full train data

# X_train = X.copy()

# y_train = y.copy()

# model_2 = XGBRegressor(n_estimators=5000, learning_rate=0.025)

# model_2.fit(X_train, y_train)



# # Make test predictions

# preds_test_2 = model_2.predict(X_test)
%%time

from sklearn.ensemble import AdaBoostRegressor



# Define and fit the model

model_3 = AdaBoostRegressor(n_estimators=3000, learning_rate=0.01)

model_3.fit(X_train, y_train)



# Get validation predictions and MAE

preds_valid_3 = model_3.predict(X_valid)

print("MAE_3 (AdaBoost Regressor): " + str(mean_absolute_error(y_valid, preds_valid_3)))
# %%time

# Define and fit model with full train data

# X_train = X.copy()

# y_train = y.copy()

# model_3 = AdaBoostRegressor(n_estimators=5000, learning_rate=0.02)

# model_3.fit(X_train, y_train)



# # Make test predictions

# preds_test_3 = model_3.predict(X_test)
%%time

from catboost import CatBoostRegressor



# Define and fit the model

model_4 = CatBoostRegressor(n_estimators=2602, learning_rate=0.022, verbose=0)

model_4.fit(X_train, y_train)



# Get validation predictions and MAE

preds_valid_4 = model_4.predict(X_valid)

print("MAE_4 (CatBoost Regressor): " + str(mean_absolute_error(y_valid, preds_valid_4)))
# %%time

# # Define and fit model with full train data

# X_train = X.copy()

# y_train = y.copy()

# model_4 = CatBoostRegressor(n_estimators=2602, learning_rate=0.022, verbose=0)

# model_4.fit(X_train, y_train)



# # Make test predictions

# preds_test_4 = model_4.predict(X_test)
# # Save test predictions to file

# output = pd.DataFrame({'Id': X_test.index,

#                        'SalePrice': preds_test_4})

# output.to_csv('submission.csv', index=False)