import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
ames_data = pd.read_csv('../input/train.csv')
ames_data.columns
ames_target = ames_data.SalePrice
ames_predictors = ames_data.drop(['SalePrice'], axis=1)
from matplotlib import pylab as plt

y = ames_target
x = ames_predictors.LotArea

plt.scatter(x, y)
plt.show()
x = ames_predictors.YearBuilt
plt.scatter(x, y)
plt.show()
ames_numeric_predictors = ames_predictors.select_dtypes(exclude=['object'])
ames_numeric_predictors.columns
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ames_numeric_predictors, ames_target, random_state=42, test_size=0.3, train_size=0.7)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return mean_absolute_error(y_test, predictions)
cols_with_missing = [col for col in X_train.columns
                                  if X_train[col].isnull().any()]
print('We drop')
print(cols_with_missing)

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)

print("MAE from dropping entire columns if any has missing:")
score_dataset(reduced_X_train, reduced_X_test, y_train, y_test)
from sklearn.impute import SimpleImputer

simple_imputer = SimpleImputer()
imputed_X_train = simple_imputer.fit_transform(X_train)
imputed_X_test = simple_imputer.transform(X_test)

print("MAE from simple imputation:")
score_dataset(imputed_X_train, imputed_X_test, y_train, y_test)
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    print(col)
    
simple_imputer = SimpleImputer()

imputed_X_train_plus = simple_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = simple_imputer.transform(imputed_X_test_plus)

print("MAE from imputation with missing fields tracking:")
score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test)
