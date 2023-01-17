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

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]



X.drop(cols_with_missing, axis=1, inplace=True)

X_test.drop(cols_with_missing, axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)
X.shape
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

cat_t = [col for col in X_train.columns if X_train[col].dtype == 'object']

drop_X_train = X_train.select_dtypes(exclude = ['object'])

drop_X_valid = X_valid.select_dtypes(exclude = ['object'])



# Check your answers

step_1.check()
# Lines below will give you a hint or solution code

step_1.hint()

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

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)



# Apply label encoder 

enc = LabelEncoder() # Your code here

for el in good_label_cols:

    label_X_train[el] = enc.fit_transform(X_train[el])

    label_X_valid[el] = enc.transform(X_valid[el])



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



# [col for col in X_train.columns if X_train[col].nunique() > 10 and 

#                             X_train[col].dtype == "object"]



# Fill in the line below: How many columns are needed to one-hot encode the 

# 'Neighborhood' variable in the training data?

num_cols_neighborhood = 25



# Check your answers

step_3.a.check()
# Lines below will give you a hint or solution code

# step_3.a.hint()

# step_3.a.solution()
# Fill in the line below: How many entries are added to the dataset by 

# replacing the column with a one-hot encoding?

OH_entries_added = 990000



# Fill in the line below: How many entries are added to the dataset by

# replacing the column with a label encoding?

label_entries_added = 0



# Check your answers

step_3.b.check()
# Lines below will give you a hint or solution code

step_3.b.hint()

#step_3.b.solution()
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
print(len([col for col in object_cols if X_train[col].nunique() < 10]))

print(20 * '#')

print(len([col for col in X_train.columns if X_train[col].nunique() < 10 and X_train[col].dtype=='object']))
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

# Use as many lines of code as you need!



One = OneHotEncoder(handle_unknown='ignore',sparse=False)



OH_train = pd.DataFrame(One.fit_transform(X_train[low_cardinality_cols]))

OH_valid = pd.DataFrame(One.transform(X_valid[low_cardinality_cols]))



### One Hot encoding removes index: put it back

OH_train.index = X_train.index

OH_valid.index = X_valid.index



### drop categorical values

num_col_train = X_train.drop(object_cols, axis=1)

num_col_valid = X_valid.drop(object_cols, axis=1)





### Concatenate the One hot ended values  with the numerical values

OH_X_train = pd.concat([num_col_train, OH_train], axis = 1) # Your code here

OH_X_valid = pd.concat([num_col_valid, OH_valid], axis = 1) # Your code here

# print(OH_X_train[:5])

print(OH_X_train.shape)

# Check your answer

step_4.check()
# Lines below will give you a hint or solution code

# step_4.hint()

# step_4.solution()
print("MAE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
m = (X_test.isnull().sum())

print(m[m>0])
obj_col = [col for col in X_test.columns if X_test[col].dtype == 'object']

num_test = X_test.select_dtypes(exclude=['object'])

# num_test.columns

# from sklearn.impute import SimpleImputer

# im = SimpleImputer()

# for el in num_test:

#     X_test[el] = pd.DataFrame(im.fit_transform(X_test[el]))

#     X_test.reshape(1, -1)

for col in num_test.columns:

    X_test[col] = X_test[col].fillna(X_test[col].mean())



print(X_test.isnull().sum())

# num_test1 = pd.DataFrame(im.fit_transform(num_test))

# num_test1.columns = num_test.columns

# num_test1.head()

m = X_test.isnull().sum()

print(m[m>0])
import matplotlib.pyplot as plt

print('Utilities',X_test['Utilities'].mode())

print('MSZoning',X_test['MSZoning'].mode())

print('Exterior1st',X_test['Exterior1st'].mode())

print('Exterior2nd',X_test['Exterior2nd'].mode())

print('KitchenQual',X_test['KitchenQual'].mode())

print('Functional',X_test['Functional'].mode())

print('SaleType',X_test['SaleType'].mode())



# X_test['MSZoning'].value_counts().plot.bar()

X_test['Utilities'].fillna('AllPub', inplace=True)

X_test['MSZoning'].fillna('RL', inplace=True)

X_test['Exterior1st'].fillna('VinylSd', inplace=True)

X_test['Exterior2nd'].fillna('VinylSd', inplace=True)

X_test['KitchenQual'].fillna('TA', inplace=True)

X_test['Functional'].fillna('Typ', inplace=True)

X_test['SaleType'].fillna('WD', inplace=True)



X_test.isna().sum()
# X_test.drop(num_test, 1, inplace=True)

# X_test = pd.concat([num_test1], axis=1)

X_test.head()
One_test = pd.DataFrame(One.transform(X_test[low_cardinality_cols]))

One_test.index = X_test.index

# print(One_test[:5])

OH_num = X_test.select_dtypes(exclude=['object'])

# # OH_num[:5]

One_hot_test = pd.concat([OH_num,One_test], axis=1)

print(One_hot_test.shape)

One_hot_test.head()
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state = 0)

model.fit(OH_X_train, y_train)

prediction = model.predict(One_hot_test)

print(prediction)

# (Optional) Your code here
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                      'SalePrice':prediction})

output.to_csv('One_hot_Submission.csv',index=False)