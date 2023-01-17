import pandas as pd

# Load data
lowa_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
print(lowa_data.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

lowa_target = lowa_data.SalePrice
lowa_predictors = lowa_data.drop(['SalePrice'], axis=1)
print(lowa_predictors.shape)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
lowa_numeric_predictors = lowa_predictors.select_dtypes(exclude=['object'])
print(lowa_numeric_predictors.shape)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(lowa_numeric_predictors, 
                                                    lowa_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
# Read the test data
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(test_data.shape)
final_test = test_data.select_dtypes(exclude=['object'])
final_test.shape
#test_X = test[data_predictors]
# Use the model to make predictions
#predicted_prices = forest_model.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.
#print(predicted_prices)

#my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
#my_submission.to_csv('submission.csv', index=False)
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
print(len(cols_with_missing))
print(X_train.shape)
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
print(reduced_X_train.shape)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print(reduced_X_test.shape)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))


# On final_test data
cols_with_missing_test = [col for col in final_test.columns 
                                 if final_test[col].isnull().any()]
print(len(cols_with_missing_test))
reduced_X_train = X_train.drop(cols_with_missing_test, axis=1)
reduced_final_test = final_test.drop(cols_with_missing_test, axis=1)

print(reduced_final_test.shape)
model = RandomForestRegressor()
model.fit(reduced_X_train, y_train)
predicted_prices = model.predict(reduced_final_test)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_handling_missing_1.csv', index=False)


from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

# On final test data

imputed_final_test = my_imputer.transform(final_test)
model = RandomForestRegressor()
model.fit(imputed_X_train, y_train)
predicted_prices = model.predict(imputed_final_test)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_handling_missing_2.csv', index=False)

# Train with full data
imputed_full_train = my_imputer.transform(lowa_numeric_predictors)
model = RandomForestRegressor()
model.fit(imputed_full_train, lowa_target)
predicted_prices = model.predict(imputed_final_test)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_handling_missing_3.csv', index=False)



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

imputed_full_train_plus = lowa_numeric_predictors.copy()
imputed_final_test_plus = final_test.copy()
cols_with_missing = [col for col in lowa_numeric_predictors.columns
                                 if lowa_numeric_predictors[col].isnull().any()]

print(len(cols_with_missing))

cols_with_missing = [col for col in final_test.columns
                                 if final_test[col].isnull().any()]

print(len(cols_with_missing))

for col in cols_with_missing:
    imputed_full_train_plus[col + '_was_missing'] = imputed_full_train_plus[col].isnull()
    imputed_final_test_plus[col + '_was_missing'] = imputed_final_test_plus[col].isnull()
    
my_imputer = Imputer()
imputed_full_train_plus = my_imputer.fit_transform(imputed_full_train_plus)
imputed_final_test_plus = my_imputer.transform(imputed_final_test_plus)
print (cols_with_missing)

model = RandomForestRegressor()
model.fit(imputed_full_train_plus, lowa_target)
predicted_prices = model.predict(imputed_final_test_plus)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_handling_missing_4.csv', index=False)
