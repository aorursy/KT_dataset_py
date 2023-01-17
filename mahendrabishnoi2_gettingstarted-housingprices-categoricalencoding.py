# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing training and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Defining predictors and target
y = train.SalePrice
X = train.drop('SalePrice', axis=1)
# Let's see a random sample of dtypes from our prediction data
print(X.dtypes.sample(10))
# Since predictors(training data) contains object data type
# we will use get_dummies function to convert these to one-hot encoding
one_hot_X = pd.get_dummies(X)
# Now we will check for NaN values if any
# Make third line as code (i.e. remove comment sign) to see if there are any NaN values
#print(one_hot_X.isnull().sum())
# Now we will apply imputaion to fill NaN values
my_imputer = SimpleImputer()
one_hot_X_imputed = my_imputer.fit_transform(one_hot_X)

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()
X_without_categoricals = X.select_dtypes(exclude=['object'])
X_without_categoricals_imputed = my_imputer.fit_transform(X_without_categoricals)

mae_without_categoricals = get_mae(X_without_categoricals_imputed, y)

mae_one_hot_encoded = get_mae(one_hot_X_imputed, y)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
# In this case one-hot-encoding does not improve the score much but 
# it is good to try it once
# Now we will do same preprocessing steps with our test data and make predictions 
one_hot_test = pd.get_dummies(test)
one_hot_X = pd.get_dummies(X)
#one_hot_test = pd.get_dummies(test)

one_hot_test_imputed = my_imputer.fit_transform(one_hot_test)
one_hot_X_imputed = my_imputer.fit_transform(one_hot_X)

# Convert imputed outputs to DataFrame
one_hot_test_imputed_df = pd.DataFrame(one_hot_test_imputed)
one_hot_X_imputed_df = pd.DataFrame(one_hot_X_imputed)

# Add columns to DataFrame
one_hot_test_imputed_df.columns = one_hot_test.columns
one_hot_X_imputed_df.columns = one_hot_X.columns

final_train, final_test = one_hot_X_imputed_df.align(one_hot_test_imputed_df,
                                                     join='left', 
                                                     axis=1)
# If there are new columns added into test data by imputation
# they contain NaN values hence fill these values
final_test = final_test.fillna(0)
# Now we will define our model, fit and predict on our model
model = RandomForestRegressor()
model.fit(final_train, y)
predictions = model.predict(final_test)
# Convert predictions and Id to a dataframe as per submission requirements
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': predictions})
submission.to_csv('submission.csv', index=False)
