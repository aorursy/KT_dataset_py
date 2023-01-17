# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sfb_permits = pd.read_csv('../input/building-permit-applications-data/Building_Permits.csv')
sfb_permits.shape
sfb_permits.info()
sfb_permits.describe()
sfb_permits.head()
missing_values = sfb_permits.isna().sum()

missing_values
total_cells = np.product(sfb_permits.shape)

total_missing = missing_values.sum()

percent_total_missing = total_missing/total_cells*100

percent_total_missing
sfb_permits.dropna()
# Dropping columns with any single NaN values.

droppped_column_with_nan = sfb_permits.dropna(axis =1)

droppped_column_with_nan.sample(9)
# Now let's just check how much data we've lost here.

print("Columns in original dataset: %d \n" % sfb_permits.shape[1])

print("Columns with na's dropped: %d" % droppped_column_with_nan.shape[1])
# Let's say we want to impute '0' wherever the data is missing(NaN)

imputed_dataset = sfb_permits.fillna(0)
#Let's check whether we have any missing values now 

imputed_dataset.isna().sum().sum()
# replace all NA's the value that comes directly before it in the same column, 

# then replace all the reamining na's with 0

using_bfill_sfb = sfb_permits.fillna(method = 'bfill', axis=0).fillna(0)

using_bfill_sfb.isnull().sum().sum()
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

using_ffill_sfb = sfb_permits.fillna(method = 'ffill', axis=0).fillna(0)

using_ffill_sfb.isnull().sum().sum()
# Let's just exclude the object data type here coz imputer can't handle it

sfb_permits_num = sfb_permits.select_dtypes(exclude ='object')

len(sfb_permits_num.dtypes)
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

data_with_imputed_values = my_imputer.fit_transform(sfb_permits_num)
from sklearn.model_selection import train_test_split



# Select target

y = sfb_permits['Permit Type']



# To keep things simple, we'll use only numerical predictors

sfb_permits_predictors = sfb_permits.drop(['Permit Type'], axis=1)

X = sfb_permits_predictors.select_dtypes(exclude=['object'])



# Divide data into training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=10, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# Score from Approach 1 (Drop Columns with Missing Values)



# Get names of columns with missing values

cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]



# Drop columns in training and validation data

reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)



print("MAE from Approach 1 (Drop columns with missing values):")

print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
# Score from Approach 2 (Imputation)

from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns



print("MAE from Approach 2 (Imputation):")

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
# Score from Approach 3 

# Make copy to avoid changing original data (when imputing)

X_train_plus = X_train.copy()

X_valid_plus = X_valid.copy()



# Make new columns indicating what will be imputed

for col in cols_with_missing:

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
# Shape of training data (num_rows, num_columns)

print(X_train.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])