# set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex3 import *

print("Setup Complete")
# load data

import pandas as pd

X = pd.read_csv('../input/train.csv', index_col='Id') 

X_test = pd.read_csv('../input/test.csv', index_col='Id')



# remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X['SalePrice']

X.drop(['SalePrice'], axis=1, inplace=True)



# to keep things simple, we'll drop columns with missing values

cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 

X.drop(cols_with_missing, axis=1, inplace=True)

X_test.drop(cols_with_missing, axis=1, inplace=True)



# break off validation set from training data

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X_train.head()
# function for comparing different approaches

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# drop columns in training and validation data

drop_X_train = X_train.select_dtypes(exclude=['object'])

drop_X_valid = X_valid.select_dtypes(exclude=['object'])



# check your answers

step_1.check()
# Lines below will give you a hint or solution code

# step_1.hint()

# step_1.solution()
print("MAE from Approach 1 (Drop categorical variables):")

print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())

print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())
# step_2.a.hint()
# check your answer (Run this code cell to receive credit!)

step_2.a.solution()
# all categorical columns

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]



# columns that can be safely label encoded

good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]

        

# problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols) - set(good_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
# drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)



# apply label encoder 

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for col in good_label_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    

# check your answer

step_2.b.check()
# Lines below will give you a hint or solution code

# step_2.b.hint()

# step_2.b.solution()
print("MAE from Approach 2 (Label Encoding):") 

print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# how many categorical variables in the training data have cardinality greater than 10?

high_cardinality_numcols = 3



# how many columns are needed to one-hot encode the 'Neighborhood' variable in the training data?

num_cols_neighborhood = 25



# check your answers

step_3.a.check()
# Lines below will give you a hint or solution code

# step_3.a.hint()

# step_3.a.solution()
# how many entries are added to the dataset by replacing the column with a one-hot encoding?

OH_entries_added = 990000



# how many entries are added to the dataset by replacing the column with a label encoding?

label_entries_added = 0



# Check your answers

step_3.b.check()
# Lines below will give you a hint or solution code

# step_3.b.hint()

# step_3.b.solution()
# columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
# use as many lines of code as you need!



# apply one-hot encoder to each column with categorical data

from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))



# one-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



# remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



# add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)



# Check your answer

step_4.check()
# Lines below will give you a hint or solution code

# step_4.hint()

# step_4.solution()
print("MAE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
# all categorical columns

test_object_cols = [col for col in X_test.columns if X_test[col].dtype == "object"]

# columns that can be safely label encoded

good_test_label_cols = [col for col in test_object_cols if set(X_train[col]) == set(X_test[col])]

# problematic columns that will be dropped from the dataset

bad_test_label_cols = list(set(test_object_cols) - set(good_test_label_cols))



# drop categorical columns that will not be encoded

label_X_test = X_test.drop(bad_test_label_cols, axis=1)



# apply label encoder

for col in good_test_label_cols:

    label_X_test[col] = label_encoder.fit_transform(X_test[col])



# missing value

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

imputed_label_X_test = pd.DataFrame(imputer.fit_transform(label_X_test))

# put column names back

imputed_label_X_test.columns = label_X_test.columns
# find columns are in both train and test

cols_in_both = [col for col in label_X_train.columns if col in imputed_label_X_test.columns]



label_X_train = label_X_train[cols_in_both]

label_X_valid = label_X_valid[cols_in_both]

imputed_label_X_test = imputed_label_X_test[cols_in_both]



# define and fit model

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(label_X_train, y_train)



# make validation prediction

preds_valid = model.predict(label_X_valid)



# make test prediction

preds_test = model.predict(imputed_label_X_test)



# save test predictions to file

output = pd.DataFrame({'Id': imputed_label_X_test.index, 'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)