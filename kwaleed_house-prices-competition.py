import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
train_data = pd.read_csv("/kaggle/input/train.csv")

pd.set_option('display.max_columns', None) #Allows to display all columns without truncation

train_data.head()
test_data = pd.read_csv("/kaggle/input/test.csv")

test_data.head()
train_data.shape
train_null_count = pd.DataFrame(train_data.isnull().sum(), columns=['nullCount'])

# Only display the columns with null values

train_null_count[train_null_count['nullCount']>0]
test_null_count = pd.DataFrame(test_data.isnull().sum(), columns=['nullCount'])

# only display the columns with null values

test_null_count[test_null_count['nullCount']>0]
# Save column names with null valeus to a list

train_null_list = train_data.columns[train_data.isnull().any()].tolist()

test_null_list = test_data.columns[test_data.isnull().any()].tolist()
# convert to set

set1=set(train_null_list)

set2=set(test_null_list)
# list of all the columns in both test and train data with null values

dropColumns = train_null_list + list(set2-set1)

len(dropColumns)
# addin id column also to the drop list

dropColumns.append('Id')

len(dropColumns)
train_data_drop = train_data.drop(columns = dropColumns)

test_data_drop = test_data.drop(columns = dropColumns)
# Shape of our training data after dropping columns

print(f"train_data_drop shape: {train_data_drop.shape}")

print(f"test_data_drop shape: {test_data_drop.shape}")
train_data_drop.isnull().sum()
test_data_drop.isnull().sum()
# Code below converts the categorical columns to one hot encoding

train_data_drop = pd.get_dummies(train_data_drop)



# Similar for test data set

X_test = pd.get_dummies(test_data_drop)
# Splits the training data into train/validation by 70-30

X_train, X_validation, y_train, y_validation = train_test_split(train_data_drop.drop(columns = ['SalePrice']), 

                                                    train_data_drop['SalePrice'], 

                                                    test_size = 0.3)
# 70% of 1460 = 1022, of the data to be used for training

print(f"X_train Shape: {X_train.shape}")



# 30% of 1460 = 438, of the data to be used for validation

print(f"X_validation Shape: {X_validation.shape}")



print(f"X_test Shape: {X_test.shape}")