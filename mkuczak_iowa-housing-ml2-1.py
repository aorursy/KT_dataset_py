import pandas as pd
iowa_data = pd.read_csv('../input/train.csv')
print(iowa_data.head())
#Making sure it's loaded in.  It is.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])
X = iowa_numeric_predictors
y = iowa_target
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y)
cols_with_missing = [col for col in train_X.columns 
                                 if train_X[col].isnull().any()]
reduced_X_train = train_X.drop(cols_with_missing, axis=1)
reduced_X_test  = test_X.drop(cols_with_missing, axis=1)
#Need this a few times, so lets def it.
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, train_y, test_y))
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(train_X)
imputed_X_test = my_imputer.transform(test_X)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, train_y, test_y))
#It's worse.
imputed_X_train_plus = train_X.copy()
imputed_X_test_plus = test_X.copy()

cols_with_missing = (col for col in train_X.columns 
                                 if train_X[col].isnull().any())
for col in cols_with_missing:
    print(col)
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, train_y, test_y))