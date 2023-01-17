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

X_test = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)



# To keep things simple, we'll drop columns with missing values

# cols_with_missing = [col for col in X.columns if X[col].isnull().any()] # original suggestion

cols_with_nans = [col for col in X.columns if X[col].hasnans] # my alternative

X.drop(cols_with_nans, axis=1, inplace=True)

X_test.drop(cols_with_nans, axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X_train.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# Fill in the lines below: drop columns in training and validation data

drop_X_train = X_train.select_dtypes(exclude=['object'])

drop_X_valid = X_valid.select_dtypes(exclude=['object'])



# Check your answers

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
print("MAE from Approach 1 (Drop categorical variables):")

print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())

print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())
#step_2.a.hint()
# Check your answer (Run this code cell to receive credit!)

step_2.a.solution()
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

dropped_X_train = X_train.drop(bad_label_cols, axis=1)

dropped_X_valid = X_valid.drop(bad_label_cols, axis=1)



# Apply label encoder 

my_encoder = LabelEncoder()



label_X_train = dropped_X_train.copy()

label_X_valid = dropped_X_valid.copy()



for col in good_label_cols:

    label_X_train[col] = my_encoder.fit_transform(dropped_X_train[col])

    label_X_valid[col] = my_encoder.transform(dropped_X_valid[col])



# Check your answer

step_2.b.check()
# Lines below will give you a hint or solution code

#step_2.b.hint()

#step_2.b.solution()
print("MAE from Approach 2 (Label Encoding):") 

print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# Fill in the line below: How many categorical variables in the training data

# have cardinality greater than 10?

high_cardinality_numcols = 3



# Fill in the line below: How many columns are needed to one-hot encode the 

# 'Neighborhood' variable in the training data?

num_cols_neighborhood = 25



# Check your answers

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution()
# Fill in the line below: How many entries are added to the dataset by 

# replacing the column with a one-hot encoding?

OH_entries_added = 99 * 10000



# Fill in the line below: How many entries are added to the dataset by

# replacing the column with a label encoding?

label_entries_added = 0



# Check your answers

step_3.b.check()
# Lines below will give you a hint or solution code

#step_3.b.hint()

#step_3.b.solution()
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
# I wanted to see some statistical characteristics of the numerical features using describe method and noticed not all features have the same value count

X_test.describe().T
# so I scanned through all the features to get a list of all that have missing values

[(col, sum(X_test[col].isnull())) for col in X_test.columns if X_test[col].hasnans == True]
# I then used a slight variation to the code above to check the types of those columns and found out that some of the categorical columns (type Object) also have missing values

[(col, X_test[col].dtype) for col in X_test.columns if X_test[col].hasnans == True]
# it was just as easy to check how many unique values each of those columns actually have

[(col, X_test[col].nunique()) for col in X_test.columns if X_test[col].hasnans == True]
# I then tested the expression to get the mean value of some of one of the float-type columns, 'BsmtFinSF1'

X_test['BsmtFinSF1'].mean()
# and used this expression to replace all the missing values in float-type columns with mean values of their respective column

[X_test[col].fillna(X_test[col].mean(), inplace=True) for col in X_test.columns if (X_test[col].hasnans == True) & (X_test[col].dtype == 'float64')]
# checking which columns with missing values remain

[(col, sum(X_test[col].isnull())) for col in X_test.columns if X_test[col].hasnans == True]
# they should all be categorical and so they are

[(col, X_test[col].dtype) for col in X_test.columns if X_test[col].hasnans == True]
# I then used the backfill method to replace each of the remaining missing values with the next occurring value in the column

[X_test[col].fillna(method='backfill', inplace=True) for col in X_test.columns if X_test[col].hasnans == True]
# the final test for columns with missing values should return an empty list as it does

[(col, X_test[col].dtype) for col in X_test.columns if X_test[col].hasnans == True]
from sklearn.preprocessing import OneHotEncoder



# Use as many lines of code as you need!

my_OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)



OH_cols_train = pd.DataFrame(my_OH_encoder.fit_transform(X_train[low_cardinality_cols]))

OH_cols_valid = pd.DataFrame(my_OH_encoder.transform(X_valid[low_cardinality_cols]))

OH_cols_test = pd.DataFrame(my_OH_encoder.transform(X_test[low_cardinality_cols]))



OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index

OH_cols_test.index = X_test.index



num_X_train = X_train.select_dtypes(exclude=['object'])

num_X_valid = X_valid.select_dtypes(exclude=['object'])

num_X_test = X_test.select_dtypes(exclude=['object'])



OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)



# Check your answer

step_4.check()
OH_X_test
# Lines below will give you a hint or solution code

#step_4.hint()

#step_4.solution()
print("MAE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
# (Optional) Your code here

X_full_train = pd.concat([OH_X_train, OH_X_valid], axis=0)

y_full_train = pd.concat([y_train, y_valid], axis=0)



model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(X_full_train, y_full_train)

preds = model.predict(OH_X_test)

preds



# a submission with this kind of formatting failed so I commented the line below and inspected the preds (the line above)

# preds.to_csv('submission.csv')
# I reformatted the preds into a Pandas Series as expected by the competition rules

output = pd.Series(preds, index=OH_X_test.index, name='SalePrice')



# and this output saved the submission 

output.to_csv('submission.csv')