import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load data
melb_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)
print(melb_predictors.head())

# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
print(melb_numeric_predictors.head())

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
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
print(cols_with_missing)
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load data
melb_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
melb_target = melb_data.Price

melb_predictors = melb_data.drop(['Price'], axis=1)
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

train_X, val_X, train_y, val_y = train_test_split(melb_numeric_predictors, melb_target, train_size=0.7, test_size=0.3, random_state=1)

imputed_train_X_plus = train_X.copy()
imputed_val_X_plus = val_X.copy()

missing_columns = [ col for col in train_X.columns if train_X[col].isnull().any() ]
print(missing_columns)

# for missing_col in missing_columns:
#     imputed_train_X_plus[missing_col + '_was_missing'] = imputed_train_X_plus[missing_col].isnull()
#     imputed_val_X_plus[missing_col + '_was_missing'] = imputed_val_X_plus[missing_col].isnull()

my_imputer = SimpleImputer()
imputed_train_X_plus = my_imputer.fit_transform(imputed_train_X_plus)
imputed_val_X_plus = my_imputer.transform(imputed_val_X_plus)

model = RandomForestRegressor(random_state=1)
model.fit(imputed_train_X_plus, train_y)

predicted_y = model.predict(imputed_val_X_plus)
mae = mean_absolute_error(predicted_y, val_y)
print('mae:' + str(mae))