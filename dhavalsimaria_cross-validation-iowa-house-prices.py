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
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def score_dataset(X, y):
    my_pipeline = make_pipeline(SimpleImputer(),RandomForestRegressor())
    scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
    return scores

print("IOWA-HOUSE-PRICES: Mean Absolute Error from Cross-Validation:")
print(score_dataset(iowa_numeric_predictors, iowa_target))
print("Mean absolute error %2f " % (-1 * score_dataset(iowa_numeric_predictors, iowa_target).mean()))
imputed_X_plus = iowa_numeric_predictors.copy()

cols_with_missing = (col for col in iowa_numeric_predictors.columns 
                                 if iowa_numeric_predictors[col].isnull().any())
for col in cols_with_missing:
    imputed_X_plus[col + '_was_missing'] = imputed_X_plus[col].isnull()

print("IOWA-HOUSE-PRICES: Mean Absolute Error from Imputation while Track What Was Imputed using PIPELINE:")
print(score_dataset(imputed_X_plus, iowa_target))
print("Mean absolute error %2f " % (-1 * score_dataset(imputed_X_plus, iowa_target).mean()))