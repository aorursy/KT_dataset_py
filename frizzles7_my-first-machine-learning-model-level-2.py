import pandas as pd

# Load data
iowa_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])
# split data into train and test

X_train, X_test, y_train, y_test = train_test_split(iowa_numeric_predictors,
                                                    iowa_target, 
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=0)
# define the score function to evaluate different approaches
# this uses a random forest

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
# identify columns with missing data

print([col for col in X_train.columns 
                   if X_train[col].isnull().any()])
# drop columns with missing data and get score

cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
# use imputation and get model score

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
# use imputation, track what was imputed, and get model score

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
# review data type of a sample of our data

iowa_predictors.dtypes.sample(10)
# use one-hot encoded categoricals (using get_dummies)

one_hot_encoded_X_train = pd.get_dummies(X_train)
one_hot_encoded_X_test = pd.get_dummies(X_test)
OHE_X_train, OHE_X_test = one_hot_encoded_X_train.align(one_hot_encoded_X_test,
                                                        join='inner', 
                                                        axis=1)
#combine OHE for objects and imputation on missing numerical values

OHE_imputed_X_train = OHE_X_train.copy()
OHE_imputed_X_test = OHE_X_test.copy()

cols_with_missing = (col for col in OHE_X_train.columns 
                                 if OHE_X_train[col].isnull().any())
for col in cols_with_missing:
    OHE_imputed_X_train[col + '_was_missing'] = OHE_imputed_X_train[col].isnull()
    OHE_imputed_X_test[col + '_was_missing'] = OHE_imputed_X_test[col].isnull()

# Imputation
my_imputer = SimpleImputer()
OHE_imputed_X_train = my_imputer.fit_transform(OHE_imputed_X_train)
OHE_imputed_X_test = my_imputer.transform(OHE_imputed_X_test)
# compare model using one-hot encoded categoricals to using numerical predictors only

from sklearn.model_selection import cross_val_score

def get_mae(X, y):
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

mae_without_categoricals = get_mae(imputed_X_train_plus, y_train)

mae_one_hot_encoded = get_mae(OHE_imputed_X_train, y_train)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
# practice with XGBoost

from xgboost import XGBRegressor

xgb_iowa_model = XGBRegressor()
xgb_iowa_model.fit(OHE_imputed_X_train, y_train, verbose=False)
# review MAE of XGB model

xgb_predictions = xgb_iowa_model.predict(OHE_imputed_X_test)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(xgb_predictions, y_test)))
# tuning - set n_estimators and early_stopping_rounds

xgb_iowa_model = XGBRegressor(n_estimators=1000)
xgb_iowa_model.fit(OHE_imputed_X_train, y_train, early_stopping_rounds=5,  
             eval_set=[(OHE_imputed_X_test, y_test)])
# tuning - set n_estimators, early_stopping_rounds, and learning rate

xgb_iowa_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_iowa_model.fit(OHE_imputed_X_train, y_train, early_stopping_rounds=5,  
             eval_set=[(OHE_imputed_X_test, y_test)])
# re-estimate model with all of the training data, using n_estimators = 81 from above

import numpy as np

#combine data
OHE_imputed_X_all = np.concatenate((OHE_imputed_X_train, OHE_imputed_X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

#re-estimate
xgb_iowa_model = XGBRegressor(n_estimators=81, learning_rate=0.05)
xgb_iowa_model.fit(OHE_imputed_X_all, y_all, verbose=False)
# predict and submit based on XGB model

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Treat the test data in the same way as training data:
# use OHE, then impute and track imputed values
one_hot_encoded_X_test_submit = pd.get_dummies(test)
OHE_X_train, OHE_X_test_submit = one_hot_encoded_X_train.align(one_hot_encoded_X_test_submit,
                                                        join='inner', 
                                                        axis=1)
OHE_imputed_X_test_submit = OHE_X_test_submit.copy()
cols_with_missing = (col for col in OHE_X_train.columns 
                                 if OHE_X_train[col].isnull().any())
for col in cols_with_missing:
    OHE_imputed_X_test_submit[col + '_was_missing'] = OHE_imputed_X_test_submit[col].isnull()
OHE_imputed_X_test_submit = my_imputer.transform(OHE_imputed_X_test_submit)

# Use the model to make predictions
predicted_prices = xgb_iowa_model.predict(OHE_imputed_X_test_submit)
# submission file

my_submission_2 = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission_2.to_csv('submission_2.csv', index=False)