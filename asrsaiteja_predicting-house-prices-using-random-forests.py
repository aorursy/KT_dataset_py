# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_file_path = '../input/train.csv'
org_train_data = pd.read_csv(train_file_path)
print('Got the data')
#data targeting & stating predictors for SalePrice
data_target = org_train_data.SalePrice
price_predictors = org_train_data.drop(['SalePrice'], axis=1)
price_numeric_predictors = price_predictors.select_dtypes(exclude=['object'])
#importing packages from sk learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#FUNCTION TO APPLY MODEL AND THEN CALCULATE MAE
X_train, X_test, y_train, y_test = train_test_split(price_numeric_predictors, data_target, train_size=0.7, test_size=0.3, random_state=0)
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)
    return mean_absolute_error(y_test, predicts)
#dropping the missing values
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
print('MAE after dropping the missing values:')
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
#modeling using imputer
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))