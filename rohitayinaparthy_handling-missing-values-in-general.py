# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
## Loading train data
train_data = pd.read_csv(main_file_path)

## Loading test data
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
## Count the occurance of missing values in each of the columns
missing_val_count_by_column = train_data.isnull().sum()

missing_val_count_by_column[missing_val_count_by_column > 0]

data_without_missing_values = train_data.dropna(axis=1) ## Drop 
## Dropping missing values in both train and test data

blank_cols = [col for col in train_data.columns if train_data[col].isnull().any()]

# Dropping columns having atleast one missing data
train_data = train_data.drop(blank_cols,axis=1)


# Dropping columns having atleast one missing data
test_data =test_data.drop(blank_cols,axis=1)
main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
## Loading train data
train_data = pd.read_csv(main_file_path)

## Loading test data
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
## Drop all categorical data and inout only numeric into SimpleImputer

train_predictors = list(train_data.dtypes[train_data.dtypes == "int64"].index)
train_predictors = train_predictors[:-1]
train_data = train_data[train_predictors]
## Imputation
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy="mean")
train_data[:] = my_imputer.fit_transform(train_data)

main_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
## Loading train data
train_data_1 = pd.read_csv(main_file_path)

## Loading test data
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_data_1.head()
## Drop all categorical data and inout only numeric into SimpleImputer

train_predictors = list(train_data_1.dtypes[train_data_1.dtypes == "int64"].index)
train_predictors = train_predictors[:-1]
train_data_1 = train_data_1[train_predictors]

#making copy of original train data
copy_train_data = train_data_1.copy()
## Make new cols indicates what will be imputed
cols_wth_missing_data = (col for col in copy_train_data.columns if copy_train_data[col].isnull().any()) 

for col in cols_wth_missing_data:
    copy_train_data[col + "_was_missing"] = copy_train_data[col].isnull()
    
# Imputation
my_imputer = SimpleImputer()
new_data = pd.DataFrame(my_imputer.fit_transform(copy_train_data))
new_data.columns = train_data_1.columns    
new_data.columns
import pandas as pd
# load the data
melb_data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
y = melb_data.Price
X = melb_data.drop(["Price"],axis=1)
## Lets collect only numeric predictors
X_numeric = X.select_dtypes(exclude=["object"])
X_train,X_test,y_train,y_test =  train_test_split(X_numeric,y,test_size = 0.20,random_state=997)
## Defined a function to calculate mae
def get_mae(train_X, train_y,test_X, test_y):
    model = RandomForestRegressor()
    model.fit(train_X,train_y)
    pred_y = model.predict(test_X)
    return mean_absolute_error(test_y,pred_y)
    

missing_cols = [col for col in X_train.columns if X_train[col].isnull().any()]
# Drop the null cols in train data
X_train_reduced = X_train.drop(missing_cols,axis= 1)

# Drop the null cols in train data
X_test_reduced = X_test.drop(missing_cols,axis= 1)

print("Mean Absolute error after dropping cols is")
print(get_mae(X_train_reduced,y_train,X_test_reduced,y_test))
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X_train_imputed = my_imputer.fit_transform(X_train)
X_test_imputed = my_imputer.fit_transform(X_test)
print("Mean Absolute error after imputation")
get_mae(X_train_imputed,y_train,X_test_imputed,y_test)
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()
for col in missing_cols:
    imputed_X_train_plus[col+ "_missing"] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col+ "_missing"] = imputed_X_test_plus[col].isnull()
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
# Fit and Transform Train data
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
# Only Transform test data as its not required for building model
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

pd.DataFrame(imputed_X_train_plus).head()
print("Mean Absolute error from imputation while track what was imputed:")
get_mae(imputed_X_train_plus,y_train,imputed_X_test_plus,y_test)

