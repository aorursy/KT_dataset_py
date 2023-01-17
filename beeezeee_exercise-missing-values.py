# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex2 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# To keep things simple, we'll use only numerical predictors

X = X_full.select_dtypes(exclude=['object'])

X_test = X_test_full.select_dtypes(exclude=['object'])



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
X_train.head()
# Shape of training data (num_rows, num_columns)

print(X_train.shape)

print(X_valid.shape)

print(X_test.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

missing_val_count_by_column = (X_valid.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

missing_val_count_by_column = (X_test.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
num_rows = 1168



num_cols_with_missing = 3





tot_missing = 212 + 6 + 58

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# Fill in the line below: get names of columns with missing values

cols_with_missing_data = [col for col in X_train.columns

                             if X_train[col].isnull().any()]# Your code here



# Fill in the lines below: drop columns in training and validation data

reduced_X_train = X_train.drop(cols_with_missing_data, axis = 1)

reduced_X_valid = X_valid.drop(cols_with_missing_data, axis = 1)



print("MAE (Drop columns with missing values):")

print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
from sklearn.impute import SimpleImputer



# Fill in the lines below: imputation

my_imputer = SimpleImputer() # Your code here

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Fill in the lines below: imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
missing_val_count_by_column = (reduced_X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])

missing_val_count_by_column = (reduced_X_valid.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])



print(reduced_X_train.shape)

print(reduced_X_valid.shape)
# Preprocessed training and validation features

final_X_train = imputed_X_train

final_X_valid = imputed_X_valid

# Imputation

final_imputer = SimpleImputer(strategy='median')

final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))

final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))



# # Imputation removed column names; put them back

final_X_train.columns = X_train.columns

final_X_valid.columns = X_valid.columns
final_X_train.index

final_X_valid.index
X_test.index
# Define and fit model

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(final_X_train, y_train)



# Get validation predictions and MAE

preds_valid = model.predict(final_X_valid)

print("MAE (Your approach):")

print(mean_absolute_error(y_valid, preds_valid))
# print(final_X_test.shape)

# missing_final_X_test = (reduced_X_test.isnull().sum())

# print(missing_final_X_test[missing_final_X_test > 0])

# missing_X_test = (X_test.isnull().sum())

# print(missing_X_test[missing_final_X_test > 0])
# Preprocess test data

# final_X_test = pd.DataFrame((final_imputer.transform(X_test)))

# final_X_test.columns = X_test.columns

# final_X_test.index = X_test.index

# Get test predictions

# preds_test = model.predict(reduced_X_train)
# Preprocess test data

final_X_test = pd.DataFrame(final_imputer.transform(X_test))

final_X_test.columns = X_test.columns

final_X_test.index = X_test.index



# Get test predictions

preds_test = model.predict(final_X_test)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)