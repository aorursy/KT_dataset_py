import pandas as pd

# Load data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train_target = train.SalePrice
train_predictors = train.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
train_numeric_predictors = train_predictors.select_dtypes(exclude=['object'])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_dev, y_train, y_dev = train_test_split(train_numeric_predictors, 
                                                  train_target,
                                                  train_size=0.7, 
                                                  test_size=0.3, 
                                                  random_state=0)

def score_dataset(X_train, X_dev, y_train, y_dev):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)
    return mean_absolute_error(y_dev, predictions)
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_dev  = X_dev.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_dev, y_train, y_dev))
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_dev = my_imputer.transform(X_dev)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_dev, y_train, y_dev))
imputed_X_train_plus = X_train.copy()
imputed_X_dev_plus = X_dev.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_dev_plus[col + '_was_missing'] = imputed_X_dev_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_dev_plus = my_imputer.transform(imputed_X_dev_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_dev_plus, y_train, y_dev))