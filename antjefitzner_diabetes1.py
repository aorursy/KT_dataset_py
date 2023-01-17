# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load training data

df_train = pd.read_csv("/kaggle/input/disease-classification-challenge/train.csv")

df_test = pd.read_csv('/kaggle/input/disease-classification-challenge/test.csv', index_col='Id')



display(df_train.head())

display(df_test.head())
# select columns

predictor_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 

                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Outcome']

# Create training predictors data

df_train = df_train[predictor_cols]





df_train.dropna(axis=0, subset=['Outcome'], inplace=True)

y = df_train['Outcome']

X = df_train.drop(['Outcome'], axis=1).select_dtypes(exclude=['object'])

train_X, valid_X, train_y, valid_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)



my_imputer = Imputer()

train_X = my_imputer.fit_transform(train_X)

valid_X = my_imputer.transform(valid_X)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



# model 1

my_model_1 = XGBRegressor()

my_model_1.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(valid_X, valid_y)], verbose=False)



# make predictions

predictions_train = my_model_1.predict(train_X)

predictions_valid = my_model_1.predict(valid_X)



# print metrics

print("Mean Absolute Error Train: " + str(mean_absolute_error(predictions_train, train_y)))

print("Mean Absolute Error Test: " + str(mean_absolute_error(predictions_valid, valid_y)))
# model 2

my_model_2 = XGBRegressor(n_estimators=1000,learning_rate=0.05)

my_model_2.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(valid_X, valid_y)], verbose=False)



# make predictions

predictions_train = my_model_2.predict(train_X)

predictions_valid = my_model_2.predict(valid_X)



# print metrics

print("Mean Absolute Error Train: " + str(mean_absolute_error(predictions_train, train_y)))

print("Mean Absolute Error Test: " + str(mean_absolute_error(predictions_valid, valid_y)))
# model 3

my_model_3 = XGBRegressor(n_estimators=1000,learning_rate=0.01)

my_model_3.fit(train_X, train_y, early_stopping_rounds=5, 

              eval_set=[(valid_X, valid_y)], verbose=False)



# make predictions

predictions_train = my_model_3.predict(train_X)

predictions_valid = my_model_3.predict(valid_X)



# print metrics

print("Mean Absolute Error Train: " + str(mean_absolute_error(predictions_train, train_y)))

print("Mean Absolute Error Test: " + str(mean_absolute_error(predictions_valid, valid_y)))

# model 4

my_model_4 = XGBRegressor(n_estimators=1000,learning_rate=0.01)

my_model_4.fit(train_X, train_y, early_stopping_rounds=5, 

              eval_set=[(valid_X, valid_y)], verbose=False,

              eval_metric = ["auc","mae"])



# make predictions

predictions_train = my_model_4.predict(train_X)

predictions_valid = my_model_4.predict(valid_X)



# print metrics

print("Mean Absolute Error Train: " + str(mean_absolute_error(predictions_train, train_y)))

print("Mean Absolute Error Test: " + str(mean_absolute_error(predictions_valid, valid_y)))

# model 5

# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

my_model_5 = XGBRegressor(n_estimators=1000,

                          learning_rate=0.01, 

                          colsample_bytree=0.8,

                          objective='binary:logistic',

                          max_depth = 7

                         )

my_model_5.fit(train_X, train_y, early_stopping_rounds=5, 

              eval_set=[(valid_X, valid_y)], verbose=False,

              eval_metric = ["auc","mae"])



# make predictions

predictions_train = my_model_5.predict(train_X)

predictions_valid = my_model_5.predict(valid_X)



# print metrics

print("Mean Absolute Error Train: " + str(mean_absolute_error(predictions_train, train_y)))

print("Mean Absolute Error Test: " + str(mean_absolute_error(predictions_valid, valid_y)))

display(train_X)
# Treat the test data in the same way as training data. In this case, pull same columns.

predictor_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 

                  'Insulin', 'BMI', 'DiabetesPedigreeFunction']

# Create training predictors data

df_test = df_test[predictor_cols]

test_X = df_test.as_matrix()

# Use the model to make predictions

predicted_submission = my_model_5.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

#print(predicted_submission)

df_test['Id'] = df_test.index





#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'Id':df_test['Id'],'Outcome':predicted_submission})



#Visualize the first 5 rows

submission.head()



#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = '../Diabetes Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)