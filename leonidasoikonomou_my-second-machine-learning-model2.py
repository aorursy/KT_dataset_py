import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor


melbourne_data_location = "../input/train.csv"
melbourne_data = pd.read_csv(melbourne_data_location)
iowa_data_location = "../input/test.csv"
iowa_data = pd.read_csv(iowa_data_location)

target_iowa_location = "../input/sample_submission.csv"
target_iowa = pd.read_csv(target_iowa_location)

#print (iowa_data.head())
#print (melbourne_data.head())
melb_target = melbourne_data.SalePrice
melb_predictors = melbourne_data.drop(['SalePrice'], axis=1)
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
#print (melb_numeric_predictors.head())
iowa_target = target_iowa.SalePrice
iowa_predictors = iowa_data
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])
imputed_melb_numeric_predictors= my_imputer.fit_transform(melb_numeric_predictors)
imputed_iowa_numeric_predictors= my_imputer.transform(iowa_numeric_predictors)
# make copy to avoid changing original data (when Imputing)

new_data = melb_numeric_predictors.copy()

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

# Imputation
my_imputer = Imputer()
new_data = my_imputer.fit_transform(new_data)
iowa_new_data = iowa_numeric_predictors.copy()

# make new columns indicating what will be imputed
cols_with_missing = (col for col in iowa_new_data.columns 
                                 if iowa_new_data[col].isnull().any())
for col in cols_with_missing:
    iowa_new_data[col + '_was_missing'] = iowa_new_data[col].isnull()

# Imputation
my_imputer = Imputer()
iowa_new_data = my_imputer.fit_transform(iowa_new_data)
train_X, val_X, train_y, val_y = train_test_split(new_data, melb_target,random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(new_data,melb_target)
val_predictions = melbourne_model.predict(val_X)
print(melbourne_model.predict(new_data))
print(mean_absolute_error(val_y, val_predictions))
train_X, val_X, train_y, val_y = train_test_split(iowa_new_data, iowa_target,random_state = 0)
iowa_model = DecisionTreeRegressor()
iowa_model.fit(iowa_new_data,iowa_target)
val_predictions = iowa_model.predict(val_X)
print(iowa_model.predict(iowa_new_data))
print(mean_absolute_error(val_y, val_predictions))
X_train, X_test, y_train, y_test = train_test_split(iowa_new_data, 
                                                    iowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(iowa_new_data, iowa_target)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

print("Mean Absolute Error from Imputation:")
print(score_dataset(new_data, iowa_new_data, melb_target, iowa_target))

predicted_prices = iowa_model.predict(iowa_new_data)

print (predicted_prices)

my_submission = pd.DataFrame({'Id': target_iowa.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)




