# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# save filepath to variable for easier access
file_path = '../input/train.csv'

# read the train data and store it in DataFrame titled melbourne_data
train_data = pd.read_csv(file_path)

# read the test data
test_data = pd.read_csv('../input/test.csv')
# print a summary of the data in Melbourne data
train_data.describe()
train_predictors= train_data.iloc[:, :-2]
test_predictors = test_data.iloc[:,:-2]
target = train_data['SalePrice']
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
# now train_predictors have category variables, use ONE to transform categorical varibales
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

# do the same for testing dataset
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='inner', 
                                                                    axis=1)
final_test.info()
# pull data into target(y) and predictors (X)
y_train = train_data['SalePrice']

imputed_X_train_plus = final_train.copy()

cols_with_missing = (col for col in final_train.columns
                        if final_train[col].isnull().any())

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)

# create model 
my_model = RandomForestRegressor()

# fit the model 
my_model.fit(imputed_X_train_plus, y_train)


# input Imputer to handle missing value 
imputed_X_test_plus = final_test.copy()

from sklearn.preprocessing import Imputer

cols_with_missing = (col for col in final_train.columns
                        if final_train[col].isnull().any())

for col in cols_with_missing:
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    
my_imputer = Imputer()
imputed_X_test_plus = my_imputer.fit_transform(imputed_X_test_plus)

# make prediction using the model we just built
predicted_prices = my_model.predict(imputed_X_test_plus)

# take a look at the predicted prices to ensure we have something sensible. 
print(predicted_prices)
# preparing file for submission

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})

# export the submission file
my_submission.to_csv('submission.csv', index = False)
