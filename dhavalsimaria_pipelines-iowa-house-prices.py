import pandas as pd

# Load data
iowa_data = pd.read_csv('../input/iowa-house-prices/train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(iowa_numeric_predictors, 
                                                    iowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    my_pipeline = make_pipeline(SimpleImputer(),RandomForestRegressor())
    my_pipeline.fit(X_train, y_train)
    preds = my_pipeline.predict(X_test)
    return mean_absolute_error(y_test, preds)

print("IOWA-HOUSE-PRICES: Mean Absolute Error from Imputation & RandomForest using Pipeline:")
print(score_dataset(X_train, X_test, y_train, y_test))
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

print("IOWA-HOUSE-PRICES: Mean Absolute Error from Imputation while Track What Was Imputed using PIPELINE:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))