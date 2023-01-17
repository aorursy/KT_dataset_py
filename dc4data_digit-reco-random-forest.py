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

print(train.columns[784])



train_backup = train

train_predictors = train.drop('label', axis=1)



# Import module for Random forest

import sklearn.ensemble



# Select predictors

predictors = train_predictors  # change this



# Converting the predictor and putcome to numpy array

x_train = train_predictors.values

y_train = train['label'].values



# Model building

model = sklearn.ensemble.RandomForestClassifier()

model.fit(x_train, y_train)



# Converting the predictor and putcome to numpy array

x_test = test.values



# Predicted output

predicted = model.predict(x_test)



# Reverse encoding for predicted outcome

# predicted = number.inverse_transform(predicted)



# Store it to a test dataset

test['Label'] = predicted

test['ImageId'] = test.index

print(test.head(5))



print(type(test))

print(test.head(5))

print(test.columns)

print(len(test.columns))

slice2 = test[['ImageId','Label']]

slice2.to_csv("RF.csv")