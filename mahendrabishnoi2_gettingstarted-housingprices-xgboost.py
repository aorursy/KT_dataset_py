# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# A function to preprocess the training and test data
# Converts dtype: 'object' to one-hot-encoding
# Applies imputation to train and test data
# Uses align function to equate columns in training and test data
# Returns final_train and final_test

def preprocess_data(train, test):
    y = train.SalePrice
    X = train.drop('SalePrice', axis=1)
    
    #convert object dtype to one-hot-encoding
    one_hot_X = pd.get_dummies(X)
    one_hot_test = pd.get_dummies(test)
    
    #Apply imputation for NaN values
    my_imputer = SimpleImputer()
    one_hot_X_imputed = my_imputer.fit_transform(one_hot_X)
    one_hot_test_imputed = my_imputer.fit_transform(one_hot_test)

    # Convert imputed outputs to DataFrame
    one_hot_X_imputed_df = pd.DataFrame(one_hot_X_imputed)
    one_hot_test_imputed_df = pd.DataFrame(one_hot_test_imputed)
    
    # Add columns to DataFrame
    one_hot_X_imputed_df.columns = one_hot_X.columns
    one_hot_test_imputed_df.columns = one_hot_test.columns
    
    # Use align to equate no. of features in training data and test data
    final_train, final_test = one_hot_X_imputed_df.align(one_hot_test_imputed_df,
                                                         join='left', 
                                                         axis=1)
    
    # If there are new columns added into test data by align
    # they contain NaN values hence fill these values
    final_test = final_test.fillna(0)
    
    return final_train, final_test
final_train, final_test = preprocess_data(train, test)
y = y = train.SalePrice
X = final_train
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
my_model = XGBRegressor(n_estimators=5000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=10, 
             eval_set=[(test_X, test_y)], verbose=False)
# make predictions
predictions = my_model.predict(final_test)
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': predictions})
submission.to_csv('submission.csv', index=False)
