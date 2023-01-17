# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.preprocessing import OneHotEncoder

from scipy import stats

from IPython.display import display

from xgboost import XGBRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import training data set

file_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'

home_data = pd.read_csv(file_path, index_col='Id')



home_data.head()
num_features = [col for col in home_data.drop(columns='SalePrice').select_dtypes(exclude='object')]

cat_features = [col for col in home_data.drop(columns='SalePrice').select_dtypes(include='object')]



# Create target and features object

y = home_data.SalePrice



X = home_data.drop(columns='SalePrice')



# Split into validation and training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)
#Get names of columns with missing values

cat_missing_cols = [col for col in X_train[cat_features].columns if X_train[col].isnull().any()]



print(cat_missing_cols)
# Simple imputation for categorical features

cat_imputer = SimpleImputer(strategy='most_frequent')



X_train_cat = X_train[cat_features].copy()

X_valid_cat = X_valid[cat_features].copy()



X_train_cat[cat_features] = cat_imputer.fit_transform(X_train[cat_features])

X_valid_cat[cat_features] = cat_imputer.transform(X_valid[cat_features])
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train_cat[col].nunique(), cat_features))

d = dict(zip(cat_features, object_nunique))



# Print number of unique entries by column, in ascending order

print(sorted(d.items(), key=lambda x: x[1]))
# OneHotEncoding or dummy encoding for categorical variables with number of unique values less than 10

low_cardinality = [col for col in cat_features if X_train_cat[col].nunique() <= 10]



# Merge imputed training and test categorical columns to get dummies for all data value in both data sets

X_cat_imputed = pd.concat([X_train_cat, X_valid_cat])



X_train_cat_LC = pd.get_dummies(X_cat_imputed[low_cardinality]).loc[X_train.index, :]

X_valid_cat_LC = pd.get_dummies(X_cat_imputed[low_cardinality]).loc[X_valid.index, :]
# Frequency ratio for categorical variables with number of unique values greater than 10

high_cardinality = list(set(cat_features)-set(low_cardinality))



X_train_cat_HC = pd.DataFrame(index=X_train.index)

X_valid_cat_HC = pd.DataFrame(index=X_valid.index)



for col in high_cardinality:

    X_train_cat_HC[col] = X_train_cat.groupby(col)[col].transform('count')/X_train_cat.shape[0]

    d = dict(zip(X_train_cat[col], X_train_cat_HC[col]))

    for i in list(set(X_valid_cat[col].unique()) - set(X_train_cat[col].unique())):

        d[i] = 0

    X_valid_cat_HC[col] = X_valid_cat[col].replace(d)
X_train_cat_prep = pd.concat([X_train_cat_HC, X_train_cat_LC], axis=1)

X_valid_cat_prep = pd.concat([X_valid_cat_HC, X_valid_cat_LC], axis=1)
#Get names of columns with missing values

num_missing_cols = [col for col in X_train[num_features].columns if X_train[col].isnull().any()]



print(num_missing_cols)
# Simple imputation numerical features

simple_imputer = SimpleImputer()



X_train_num_S1 = X_train[num_features].copy()

X_valid_num_S1 = X_valid[num_features].copy()



X_train_num_S1[num_features] = simple_imputer.fit_transform(X_train[num_features])

X_valid_num_S1[num_features] = simple_imputer.transform(X_valid[num_features])
# Merge categorical and numerical features

X_train_S1 = pd.concat([X_train_cat_prep, X_train_num_S1], axis=1)

X_valid_S1 = pd.concat([X_valid_cat_prep, X_valid_num_S1], axis=1)
# KNN imputation for numerical features

KNN_imputer = KNNImputer()



X_train_num_S2 = X_train[num_features].copy()

X_valid_num_S2 = X_valid[num_features].copy()



X_train_num_S2[num_features] = KNN_imputer.fit_transform(X_train[num_features])

X_valid_num_S2[num_features] = KNN_imputer.transform(X_valid[num_features])
# Merge categorical and numerical features

X_train_S2 = pd.concat([X_train_cat_prep, X_train_num_S2], axis=1)

X_valid_S2 = pd.concat([X_valid_cat_prep, X_valid_num_S2], axis=1)
# Define the model, and tune params using RandomizedSearchCV



rf_model = RandomForestRegressor(random_state=1)

n_estimators_grid = {'n_estimators': stats.randint(100, 300)}



model_CV_S1_rf = RandomizedSearchCV(rf_model, param_distributions=n_estimators_grid,

                                    scoring='neg_root_mean_squared_error', n_iter=25, random_state=1, n_jobs=-1)

model_CV_S1_rf.fit(X_train_S1, y_train)



# Best score and parameter with RandomizedSearchCV

best_param_S1_rf = model_CV_S1_rf.best_params_

print('Best parameter: {}'.format(best_param_S1_rf))

print('Best score: {}'.format(model_CV_S1_rf.best_score_))



val_predictions = model_CV_S1_rf.predict(X_valid_S1)

val_mae = mean_squared_error(y_true=y_valid, y_pred=val_predictions, squared=False)



print("Validation MAE for the best Random Forest model with Strategy 1: {:,.0f}".format(val_mae))
XGB_model_S1 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

XGB_model_S1.fit(X_train_S1, y_train, early_stopping_rounds=5,

              eval_set=[(X_valid_S1, y_valid)], verbose=False)



# Best score and parameter with RandomizedSearchCV

best_param_S1_xgb = XGB_model_S1.best_iteration

print('Best parameter: {}'.format(best_param_S1_xgb))

print('Best score: {}'.format(XGB_model_S1.best_score))



val_predictions = XGB_model_S1.predict(X_valid_S1)

val_mae = mean_squared_error(y_true=y_valid, y_pred=val_predictions, squared=False)



print("Validation MAE for the best XGBoost model with Strategy 1: {:,.0f}".format(val_mae))
# Define the model, and tune params using RandomizedSearchCV



rf_model = RandomForestRegressor(random_state=1)

n_estimators_grid = {'n_estimators': stats.randint(100, 300)}



model_CV_S2 = RandomizedSearchCV(rf_model, param_distributions=n_estimators_grid,

                                 scoring='neg_root_mean_squared_error', n_iter=25, random_state=1, n_jobs=-1)

model_CV_S2.fit(X_train_S2, y_train)



# Best score and parameter with RandomizedSearchCV

best_param_S2 = model_CV_S2.best_params_

print('Best parameter: {}'.format(best_param_S2))

print('Best score: {}'.format(model_CV_S2.best_score_))



val_predictions = model_CV_S2.predict(X_valid_S2)

val_mae = mean_squared_error(y_true=y_valid, y_pred=val_predictions, squared=False)



print("Validation MAE for the best Random Forest model with Strategy 2: {:,.0f}".format(val_mae))
XGB_model_S2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

XGB_model_S2.fit(X_train_S2, y_train, early_stopping_rounds=5,

              eval_set=[(X_valid_S2, y_valid)], verbose=False)



# Best score and parameter with RandomizedSearchCV

best_param_S2_xgb = XGB_model_S2.best_iteration

print('Best parameter: {}'.format(best_param_S2_xgb))

print('Best score: {}'.format(XGB_model_S2.best_score))



val_predictions = XGB_model_S2.predict(X_valid_S2)

val_mae = mean_squared_error(y_true=y_valid, y_pred=val_predictions, squared=False)



print("Validation MAE for the best XGBoost model with Strategy 2: {:,.0f}".format(val_mae))
# Best model

best_model = XGB_model_S2
# Import test data

test_data_path = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

test_data = pd.read_csv(test_data_path, index_col='Id')
# Simple imputation for categorical features

cat_imputer = SimpleImputer(strategy='most_frequent')



X_cat = X[cat_features].copy()

X_test_cat = test_data[cat_features].copy()



X_cat[cat_features] = cat_imputer.fit_transform(X[cat_features])

X_test_cat[cat_features] = cat_imputer.transform(test_data[cat_features])



# OneHotEncoding or dummy encoding for categorical variables with number of unique values less than 10

# Merge imputed training and test categorical columns to get dummies for all data value in both data sets

X_cat_imputed = pd.concat([X_cat, X_test_cat])



X_cat_LC = pd.get_dummies(X_cat_imputed[low_cardinality]).loc[X.index, :]

X_test_cat_LC = pd.get_dummies(X_cat_imputed[low_cardinality]).loc[test_data.index, :]



# Frequency ratio for categorical variables with number of unique values greater than 10

X_cat_HC = pd.DataFrame(index=X.index)

X_test_cat_HC = pd.DataFrame(index=test_data.index)



for col in high_cardinality:

    X_cat_HC[col] = X_cat.groupby(col)[col].transform('count')/X_cat.shape[0]

    d = dict(zip(X_cat[col], X_cat_HC[col]))

    for i in list(set(X_test_cat[col].unique()) - set(X_cat[col].unique())):

        d[i] = 0

    X_test_cat_HC[col] = X_test_cat[col].replace(d)

    

X_cat = pd.concat([X_cat_HC, X_cat_LC], axis=1)

X_test_cat = pd.concat([X_test_cat_HC, X_test_cat_LC], axis=1)
"""

# Strategy 1

# Simple imputation numerical features

simple_imputer = SimpleImputer()



X_num = X[num_features].copy()

X_test_num = test_data[num_features].copy()



X_num[num_features] = simple_imputer.fit_transform(X[num_features])

X_test_num[num_features] = simple_imputer.transform(test_data[num_features])



# Merge categorical and numerical features

X = pd.concat([X_cat, X_num], axis=1)

X_test = pd.concat([X_test_cat, X_test_num], axis=1)

"""
# Strategy 2

# KNN imputation numerical features

KNN_imputer = KNNImputer()



X_num = X[num_features].copy()

X_test_num = test_data[num_features].copy()



X_num[num_features] = KNN_imputer.fit_transform(X[num_features])

X_test_num[num_features] = KNN_imputer.transform(test_data[num_features])



# Merge categorical and numerical features

X = pd.concat([X_cat, X_num], axis=1)

X_test = pd.concat([X_test_cat, X_test_num], axis=1)
"""

# Train best model on full training data

rf_model = RandomForestRegressor(n_estimators=best_param_random['n_estimators'], random_state=1)

rf_model.fit(X, y)



# make predictions which we will submit

test_preds = rf_model.predict(X_test)

"""
# Train best model

XGB_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

XGB_model.fit(X_train_S2, y_train, early_stopping_rounds=10,

              eval_set=[(X_valid_S2, y_valid)], verbose=False)



# make predictions which we will submit

test_preds = XGB_model.predict(X_test)
# Save predictions in format used for competition scoring



output = pd.DataFrame({'Id': test_data.index,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)