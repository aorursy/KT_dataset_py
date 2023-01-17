import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
lowa_train_data = pd.read_csv('../input/train.csv')
lowa_test_data = pd.read_csv('../input/test.csv')

#select only numeric columns for predictions
train_y = lowa_train_data.SalePrice

lowa_predictors = lowa_train_data.drop(['SalePrice'], axis=1)
lowa_numeric_predictors = lowa_predictors.select_dtypes(exclude=['object'])
print(lowa_numeric_predictors.columns)
lowa_numeric_predictors
train_X, val_X, train_y, val_y = train_test_split(lowa_numeric_predictors, train_y, random_state=42)
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_predictions = model.predict(X_test)
    return mean_absolute_error(y_test, y_predictions)
cols_with_missing_values = [col for col in train_X.columns if train_X[col].isnull().any()]
#Drop these columns
reduced_train_X = train_X.drop(cols_with_missing_values, axis=1)
reduced_val_X = val_X.drop(cols_with_missing_values, axis=1)
print("Mean absolute error after droping columns with missing values")
print(score_dataset(reduced_train_X, reduced_val_X, train_y, val_y))
imputer = Imputer()
imputed_train_X = imputer.fit_transform(train_X)
imputed_val_X = imputer.transform(val_X)
print("Mean absolute error after imputation")
print(score_dataset(imputed_train_X, imputed_val_X, train_y, val_y))
imputed_train_X_plus = train_X.copy()
imputed_val_X_plus = val_X.copy()
missing_columns = [col for col in imputed_train_X_plus.columns if imputed_train_X_plus[col].isnull().any()]
for col in missing_columns:
    imputed_train_X_plus[col+ '_was_missing'] = imputed_train_X_plus[col].isnull()
    imputed_val_X_plus[col+'_was_missing'] = imputed_val_X_plus[col].isnull()

#Imputation
imputer = Imputer()
imputed_train_X_plus = imputer.fit_transform(imputed_train_X_plus)
imputed_val_X_plus = imputer.transform(imputed_val_X_plus)
print("Mean absolute error after imputation while track what was imputed")
print(score_dataset(imputed_train_X_plus, imputed_val_X_plus, train_y, val_y))


print(lowa_predictors.dtypes.sample(10))
lowa_predictors
from sklearn.model_selection import cross_val_score

def get_mae(X, y):
    return  -1*cross_val_score(RandomForestRegressor(50), X, y, scoring='neg_mean_absolute_error').mean()
columns_with_missing_values = [col for col in lowa_predictors.columns if lowa_predictors[col].isnull().any() and lowa_predictors[col].dtype == 'object']
lowa_predictors_without_missing_categorical = lowa_predictors.drop(columns_with_missing_values, axis=1)
lowa_test_data_without_missing_categorical = lowa_test_data.drop(columns_with_missing_values, axis=1)
one_hot_encoded_training_predictors = pd.get_dummies(lowa_predictors_without_missing_categorical)
one_hot_encoded_test_predictors = pd.get_dummies(lowa_test_data_without_missing_categorical)

one_hot_encoded_training_predictors, one_hot_encoded_test_predictors =one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join='left', axis=1)
imputer = Imputer()
imputed_one_hot_encoded_training_predictors = imputer.fit_transform(one_hot_encoded_training_predictors)
target = lowa_train_data.SalePrice
print("MAE with one hot encoded training predictors {}".format(get_mae(imputed_one_hot_encoded_training_predictors, target)))

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
test_X = test_data.select_dtypes(exclude=['object'])
imputer = Imputer()
imputed_X = imputer.fit_transform(X)
imputed_test_X = imputer.transform(test_X)

train_X, val_X, train_y, val_y = train_test_split(imputed_X, y.as_matrix(), test_size=0.25)
train_y = train_y.reshape((train_y.shape[0], 1))
val_y = val_y.reshape((val_y.shape[0], 1))
model = XGBRegressor()
model.fit(train_X, train_y, verbose=False)
predictions = model.predict(val_X)
print("Mean absolute error is {}".format(mean_absolute_error(predictions, val_y)))
model = XGBRegressor(n_estimators=1000, learning_rate=0.01)
model.fit(train_X, train_y, eval_set=[(val_X, val_y)], early_stopping_rounds=5, verbose=False)
predictions = model.predict(val_X)
print("Mean absolute error is {}".format(mean_absolute_error(predictions, val_y)))
model.fit(imputed_X, y.as_matrix(), verbose=False)
X.columns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

model = GradientBoostingRegressor()
model.fit(imputed_X, y.as_matrix())
my_plot = plot_partial_dependence(model, features=[3, 4, 16], X = imputed_X, feature_names = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold'],
                                  grid_resolution=10)
from sklearn.pipeline import make_pipeline
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000))
my_pipeline.fit(X, y)
predictions = my_pipeline.predict(test_X)
print(predictions[:5])

from sklearn.model_selection import cross_val_score
-1*cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error').mean()
