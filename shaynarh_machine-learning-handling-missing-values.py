import pandas as pd
iowa_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
iowa_data.columns
#set up train variables unly using numeric predictors

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis = 1)
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])

train_X, val_X, train_y, val_y = train_test_split(iowa_numeric_predictors, iowa_target, random_state = 0)
#define score_dataset, will be used to compare different methods of imputing later by using a 
#random forest for all of them, and comparing the MAEs

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
#drop all columns with missing values (with only numerical values as predictors)

cols_with_missing = [col for col in train_X.columns 
                                 if train_X[col].isnull().any()]
reduced_X_train = train_X.drop(cols_with_missing, axis=1)
reduced_X_test  = val_X.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, train_y, val_y))
#drop all columns with missing values (with both numeric and categorical variables)

cols_with_missing = [col for col in train_X.columns 
                                 if train_X[col].isnull().any()]
reduced_X_train = train_X.drop(cols_with_missing, axis=1)
reduced_X_test  = val_X.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, train_y, val_y))

#use imputation

from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(train_X)
imputed_X_test = my_imputer.transform(val_X)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, train_y, val_y))
#use imputation, but also print which columns were missing values

imputed_X_train_plus = train_X.copy()
imputed_X_test_plus = val_X.copy()

cols_with_missing = (col for col in train_X.columns 
                                 if train_X[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, train_y, val_y))
#to hot-encode with dummies for categorical variables:

#one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
#one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
#final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    #join='left', 
                                                                    #axis=1)
