import pandas as pd

# Load data
path = '../input/melbourne-housing-snapshot/melb_data.csv'
melb_data = pd.read_csv(path)
melb_data.info()
# Because we predict house price -> target: Price
melb_target = melb_data.Price
melb_predictors = melb_data.drop(columns='Price')
melb_predictors.info()
# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude='object')
melb_numeric_predictors.dtypes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors,
                                                   melb_target,
                                                   train_size = 0.7,
                                                   random_state = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Definite function to determine error of model
def score_dataset(X_train, X_test, y_train, y_test,
                  model_Class = RandomForestRegressor, metric_error = mean_absolute_error):
    model = model_Class()
    model.fit(X_train, y_train)
    pred_results = model.predict(X_test)
    return metric_error(y_test, pred_results)
# determine columns have missing value (NAN)
cols_with_missing = [col for col in X_train.columns
                                    if X_train[col].isnull().any()]
X_train_reduced = X_train.drop(columns=cols_with_missing)
X_test_reduced = X_test.drop(columns=cols_with_missing)
print ("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(X_train=X_train_reduced,
                   y_train=y_train,
                   X_test= X_test_reduced,
                   y_test= y_test))
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X_train_imputed = my_imputer.fit_transform(X_train)
X_test_imputed = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(X_train_imputed, X_test_imputed, y_train, y_test))
X_train_imputed_plus = X_train.copy()
X_test_imputed_plus = X_test.copy()
cols_with_missing = [col for col in X_train.columns
                                    if X_train[col].isnull().any()]
cols_with_missing
for col in cols_with_missing:
    X_train_imputed_plus[col + '_was_missing'] = X_train_imputed_plus[col].isnull()
    X_test_imputed_plus[col + '_was_missing'] = X_test_imputed_plus[col].isnull()
# Imputation
my_imputer = SimpleImputer()
X_train_imputed_plus = my_imputer.fit_transform(X_train_imputed_plus)
X_test_imputed_plus = my_imputer.transform(X_test_imputed_plus)
print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(X_train_imputed_plus, X_test_imputed_plus, y_train, y_test))