# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split



# Read the data

X = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id') 

X_test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')



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

X_train.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# All categorical columns

object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
nullCol=[col for col in X_test.columns if X_test[col].isnull().any()]

for col in nullCol:

    if X_test[col].dtype=='object':

        most_frequent=max(set(list(X_train[col])),key=list(X_train[col]).count)

        X_test[col]=X_test[col].fillna(most_frequent)

    else:

        X_test[col]=X_test[col].fillna(X_train[col].median())

        
# Apply one-hot encoder to each column with categorical data

from sklearn.preprocessing import OneHotEncoder



OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index

OH_cols_test.index = X_test.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)

num_X_test = X_test.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(OH_X_train, y_train)



# Get validation predictions and MAE

preds_valid = model.predict(OH_X_valid)

print("MAE (Your approach):")

print(mean_absolute_error(y_valid, preds_valid))

preds_test = model.predict(OH_X_test)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)