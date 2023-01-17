# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

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



# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Fill in the line below: How many rows are in the training data?

num_rows = 1168



# Fill in the line below: How many columns in the training data

# have missing values?

num_cols_with_missing = 3



# Fill in the line below: How many missing entries are contained in 

# all of the training data?

tot_missing = 276



# Check your answers

step_1.a.check()
# Lines below will give you a hint or solution code

#step_1.a.hint()

#step_1.a.solution()
#step_1.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_1.b.solution()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# Fill in the line below: get names of columns with missing values

val_count_null =(X_train.isnull().sum())

val_count_null = val_count_null[val_count_null > 0].index



# Fill in the lines below: drop columns in training and validation data

reduced_X_train = X_train.drop(val_count_null, axis=1)

reduced_X_valid = X_valid.drop(val_count_null, axis=1)



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
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



# Check your answers

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution()
print("MAE (Imputation):")

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
#step_3.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_3.b.solution()
# Make copy to avoid changing original data (when imputing)

X_train_plus = X_train.copy()

X_valid_plus = X_valid.copy()



# Make new columns indicating what will be imputed

for col in val_count_null:

    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))



# Imputation removed column names; put them back

imputed_X_train_plus.columns = X_train_plus.columns

imputed_X_valid_plus.columns = X_valid_plus.columns



print("MAE from Approach 3 (An Extension to Imputation):")

print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))





# Preprocessed training and validation features

final_X_train = imputed_X_train_plus

final_X_valid = imputed_X_valid_plus



# Check your answers

step_4.a.check()
# Lines below will give you a hint or solution code

step_4.a.hint()

#step_4.a.solution()
# Define and fit model

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(final_X_train, y_train)



# Get validation predictions and MAE

preds_valid = model.predict(final_X_valid)

print("MAE (Your approach):")

print(mean_absolute_error(y_valid, preds_valid))
# Make copy to avoid changing original data (when imputing)

X_test_plus = X_test.copy()





# Make new columns indicating what will be imputed

for col in val_count_null:

    X_test_plus[col + '_was_missing'] = X_test_plus[col].isnull()





# Imputation

my_imputer = SimpleImputer()

imputed_X_test_plus = pd.DataFrame(my_imputer.fit_transform(X_test_plus))



# Imputation removed column names; put them back

imputed_X_test_plus.columns = X_test_plus.columns



#print("MAE from Approach 3 (An Extension to Imputation):")

#print(score_dataset(imputed_X_test_plus, imputed_X_test_plus, y_train, y_valid))





# Preprocessed training and validation features

final_X_test = imputed_X_test_plus

# Get test predictions

preds_test = preds_valid = model.predict(final_X_test)



step_4.b.check()
# Lines below will give you a hint or solution code

#step_4.b.hint()

#step_4.b.solution()
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)