import pandas as pd

# Load data
melb_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

#load Iowa data
IA_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

IA_data_target = IA_data['SalePrice']
IA_data_predictors = IA_data.drop(['SalePrice'], axis=1)
IA_data_predictors = IA_data_predictors.select_dtypes(exclude=['object'])
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
#with Iowa dataset
X_train, X_test, y_train, y_test = train_test_split(IA_data_predictors,
                                                   IA_data_target,
                                                   train_size=0.8,
                                                   test_size=0.2,
                                                   random_state=0)
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
#Get MAE with dropping columns with missing data in Iowa dataset
cols_with_missing_IA = [column for column in X_train.columns
                       if X_train[column].isnull().any()]
reduced_X_train_IA = X_train.drop(cols_with_missing_IA, axis=1)
reduced_X_test_IA = X_test.drop(cols_with_missing_IA, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train_IA, reduced_X_test_IA, y_train, y_test))
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
#using Imputer with Iowa dataset
imputed_X_train_IA = my_imputer.fit_transform(X_train)
imputed_X_test_IA = my_imputer.fit_transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train_IA, imputed_X_test_IA, y_train, y_test))
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
#with Iowa dataset
imputed_X_train_plus_IA = X_train.copy()
imputed_X_test_plus_IA = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus_IA[col + '_was_missing'] = imputed_X_train_plus_IA[col].isnull()
    print(imputed_X_train_plus_IA[col + '_was_missing'])
    imputed_X_test_plus_IA[col + '_was_missing'] = imputed_X_test_plus_IA[col].isnull()
    assert False

# Imputation
#my_imputer = Imputer()
#imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
#imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

#print("Mean Absolute Error from Imputation while Track What Was Imputed:")
#print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))