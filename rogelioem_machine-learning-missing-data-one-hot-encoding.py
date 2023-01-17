import pandas as pd

# Load data
iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
iowa_data = pd.read_csv(iowa_file_path) 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

##Separating data in feature predictors, and target to predict
iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

### Separate dataset samples into train and test data 
X_train, X_test, y_train, y_test = train_test_split(iowa_numeric_predictors, 
                                                    iowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

### Function to score or evaluate how good its the model
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
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
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

##Separating data in feature predictors, and target to predict
iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

### Separate dataset samples into train and test data 
X_train2, X_test2, y_train2, y_test2 = train_test_split(iowa_predictors, 
                                                    iowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

### Function to score or evaluate how good its the model
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

### Getting train & test datasets for model evaluations: One-Hot-Encoding & dropping categorical features.
X_train_OHE = pd.get_dummies(X_train2)
X_test_OHE = pd.get_dummies(X_test2)


X_train_without_categoricals = X_train2.select_dtypes(exclude=['object'])
X_test_without_categoricals = X_test2.select_dtypes(exclude=['object'])

### Getting aligned train & test datasets for both model evaluations
final_train, final_test = X_train_OHE.align(X_test_OHE,
                                            join='left', 
                                            axis=1)
final_train_without_categorical, final_test_without_categorical = X_train_without_categoricals.align(X_test_without_categoricals,
                                                                    join='left', 
                                                                    axis=1)

### Imputing train & test datasets, filling NaN values for both model evaluations
imputed_final_train = my_imputer.fit_transform(final_train)
imputed_final_test = my_imputer.transform(final_test)

imputed_final_train_without_categorical = my_imputer.fit_transform(final_train_without_categorical)
imputed_final_test_without_categorical = my_imputer.transform(final_test_without_categorical)

### Printing Errors for One-Hot-Encoded datasets vs Without categorical features.
print("Mean Absolute Error from Imputation&OHE:")
print(score_dataset(imputed_final_train, imputed_final_test, y_train2, y_test2))

print("Mean Absolute Error from Imputation&WithoutCategorical:")
print(score_dataset(imputed_final_train_without_categorical, imputed_final_test_without_categorical, y_train2, y_test2))


