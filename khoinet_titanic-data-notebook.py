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
train_data_path = "../input/train.csv"

test_data_path = "../input/test.csv"



train_data = pd.read_csv(train_data_path)

test_data = pd.read_csv(test_data_path)
print(train_data.head(0))

print(train_data.describe())



# Cleaning data

# train_data = train_data.dropna()

train_data = train_data.fillna(train_data.mode())

print(train_data.isna().sum())



train_data_modified = train_data.replace("male", 1)

train_data_modified = train_data_modified.replace("female", 0)

label = train_data["Survived"]
# Chosen column as features for the model

features_chosen = ["Pclass","Parch","Sex","SibSp","Fare"]



X = train_data_modified[features_chosen]

X = X.astype('float64')

label = label.astype('float64')

train_data_modified.head(3)

# Model 1 linear regression model.

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score

linear_model = LinearRegression()

linear_model = linear_model.fit(X,label)



# Model 2 LogisticRegression model

from sklearn.linear_model import LogisticRegression

logisticRegression_model = LogisticRegression()

logisticRegression_model = logisticRegression_model.fit(X,label)



# Model 2 RandomForest model

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

RandomForest_model = RandomForestClassifier(n_estimators=700)

RandomForest_model = RandomForest_model.fit(X,label)
def get_predictions(input_data, label, model):

    predictions = model.predict(input_data)

    for i in range(0, len(predictions)):

        if predictions[i] > 0.5:

            predictions[i] = 1

        else:

            predictions[i] = 0

    accuracy_rate = accuracy_score(label, predictions)

    print(accuracy_rate)

    return predictions
predictions1 = get_predictions(X, label, linear_model)

predictions2 = get_predictions(X, label, logisticRegression_model)

predictions3 = get_predictions(X, label, RandomForest_model)
## Test data cleansing

test_data_modified = test_data[features_chosen]



# Get all the required label

label_data_path = "../input/gender_submission.csv"

label_data = pd.read_csv(label_data_path)



#test_data_modified = test_data_modified.astype('float64')



test_data_modified.isnull().sum()

null_data = test_data_modified[test_data_modified.isnull().any(axis=1)]



label_data = label_data.drop('PassengerId', axis=1)



data = pd.concat([test_data_modified, label_data], axis=1)

#data = data.dropna()

data = data.fillna(data.median())



data = data.replace("male", 1)

data = data.replace("female", 0)



label_data = data['Survived']

test_data_modified = data[features_chosen].astype('float64')
result_1 = get_predictions(test_data_modified, label_data, linear_model)

result_2 = get_predictions(test_data_modified, label_data, logisticRegression_model)

result_3 = get_predictions(test_data_modified, label_data, RandomForest_model)
label_data = pd.read_csv(label_data_path)

label_data['Survived'] = result_2.astype('int64')



label_data.to_csv('submission.csv', index=False)