import pandas as pd

import numpy as np

import csv as csv

from sklearn.ensemble import RandomForestClassifier



train = pd.read_csv(open('../input/train.csv','rb'))

test = pd.read_csv(open('../input/test.csv','rb'))

print(test.head(5))

print(train.head(5))

print(test.head(5))



print(len(train.columns))



train_backup = train

train_predictors = train.drop('species', axis=1)

train_predictors = train.drop('id', axis=1)



# Import module for Random forest

import sklearn.ensemble



# Select predictors

predictors = train_predictors  # change this



# Converting the predictor and putcome to numpy array

x_train = train_predictors.values

y_train = train['species'].values



# Model building

model = sklearn.ensemble.RandomForestClassifier()

model.fit(x_train, y_train)



# Converting the predictor and putcome to numpy array

test_predictors = test.drop('id', axis=1)

x_test = test_predictors.values



# Predicted output

predicted = model.predict(x_test)



# Reverse encoding for predicted outcome

# predicted = number.inverse_transform(predicted)



# Store it to a test dataset

test['Label'] = predicted

test['id'] = test.id

print(test.head(5))



print(type(test))

print(test.head(5))

print(test.columns)
print(len(test.columns))

slice2 = test[['ImageId','Label']]

slice2.to_csv("LeafClass.csv")