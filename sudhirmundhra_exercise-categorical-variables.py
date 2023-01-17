# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex3 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X = pd.read_csv('../input/train.csv', index_col='Id') 



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)



# To keep things simple, we'll drop columns with missing values

cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 

X.drop(cols_with_missing, axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)
X_train.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
drop_X_train = X_train.select_dtypes(exclude=['object'])

drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")

print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
X_train.head(5).T
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())

print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())
# All categorical columns

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train[col]) == set(X_valid[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
from sklearn.preprocessing import LabelEncoder



# Drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)



# Apply label encoder 

label_encoder = LabelEncoder()

for col in set(good_label_cols):

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])
print("MAE from Approach 2 (Label Encoding):") 

print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# How many categorical variables in the training data have cardinality greater than 10?

high_cardinality_numcols = len([col for col in object_cols if X_train[col].nunique() > 10])



# How many columns are needed to one-hot encode the 'Neighborhood' variable in the training data?

num_cols_neighborhood = [X_train[col].nunique() for col in object_cols if col == "Neighborhood"][0]
# How many entries are added to the dataset by replacing the column with a one-hot encoding?

OH_entries_added = 1e4*100 - 1e4



# How many entries are added to the dataset by replacing the column with a label encoding?

label_entries_added = 0
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))



# One-hot encoding removed index; putting it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
print("MAE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
# Read test data

X_test = pd.read_csv('../input/test.csv', index_col='Id')



# Drop columns with missing values

X_test.drop(cols_with_missing, axis=1, inplace=True)



# Print columns with NaN 

nans = []



for cols in X_test[low_cardinality_cols].columns:

  if X_test[cols].isnull().values.any() == True:

    nans.append(cols)



print("Columns with NaN: ", nans, "\n")



# Drop rows with NaN

print("Shape before dropping rows: ", X_test.shape, "\n")

X_test.dropna(axis=0, subset=nans, inplace=True)

print("Shape after dropping rows: ", X_test.shape)



# Check columns with NaN

nans = []



for cols in X_test[low_cardinality_cols].columns:

  if X_test[cols].isnull().values.any() == True:

    #print(cols, ":-", X_test[cols].unique())

    nans.append(cols)



print("Columns with NaN", nans, "\n")



# Apply one-hot encoder to each column with categorical data

OH_cols_test = pd.DataFrame(OH_encoder.fit_transform(X_test[low_cardinality_cols]))



# One-hot encoding removed index; putting it back

OH_cols_test.index = X_test.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_test = X_test.drop(object_cols, axis=1)



# Check for columns in train, validation and test

print("Shape of numerical columns: ", num_X_train.shape[1], num_X_valid.shape[1], num_X_test.shape[1])

print("Shape of OH columns: ", OH_cols_train.shape[1], OH_cols_valid.shape[1], OH_cols_test.shape[1])

print("OH_cols_test has ",  OH_cols_train.shape[1] - OH_cols_test.shape[1], " columns less than OH_cols_train")



# Add one-hot encoded columns to numerical features

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)



# Columns not in OH_X_test as compared to OH_X_train

diff_cols = list((set(OH_X_train.columns) - set(OH_X_test.columns)))

print("Difference in OH columns between train and test", diff_cols, "\n")



# create columns with zeros

print("Difference in shapes before addition of ZERO columns", OH_X_train.shape[1], OH_X_test.shape[1], "\n")



import numpy as np

for cols in diff_cols:

  OH_X_test[cols] = np.zeros(OH_X_test.shape[0])



print("Difference in shapes after addition of ZERO columns", OH_X_train.shape[1], OH_X_test.shape[1], "\n")



print("MAE from One-Hot Encoding:") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))



# Preprocess test data

from sklearn.impute import SimpleImputer

final_imputer = SimpleImputer(strategy='median')

final_X_test = pd.DataFrame(final_imputer.fit_transform(OH_X_test))



# Get test predictions

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(OH_X_train, y_train)

preds_test = model.predict(final_X_test)

print("Predict 10 values: ", preds_test[0:10])



# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)