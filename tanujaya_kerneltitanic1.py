# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# save filepath to variable

genderPath = '../input/gender_submission.csv'

trainPath = '../input/train.csv'

testPath = '../input/test.csv'



# read data and store in dataframe

genderData = pd.read_csv(genderPath)

trainDataNotImputer = pd.read_csv(trainPath)

testDataNotImputer = pd.read_csv(testPath)
# XTestHotEnc = pd.get_dummies(XTest)

# XTestHotEnc.head()



trainDataHotKey = pd.get_dummies(trainDataNotImputer)

testDataHotKey = pd.get_dummies(testDataNotImputer)
# trainData = trainData.dropna(axis=0)

from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

trainData = my_imputer.fit_transform(trainDataHotKey)

testData = my_imputer.fit_transform(testDataHotKey)
trainData
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

X = trainData[features]

XTest = testData[features]

# describe train data

X.describe()
X.head()
# XOneHotEnc = pd.get_dummies(X)

# XOneHotEnc.head()
from sklearn.tree import DecisionTreeClassifier



#Create classifier object with default hyperparameters

model = DecisionTreeClassifier()  



#Fit our classifier using the training features and the training target values

model.fit(XOneHotEnc, y)
from sklearn.metrics import mean_absolute_error



prediction = model.predict(XOneHotEnc)

print(mean_absolute_error(y, prediction))
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y =  train_test_split(XOneHotEnc, y, random_state = 0)

model.fit(train_X,train_y)



val_predictions = model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
XTest.describe()
# XTestHotEnc = pd.get_dummies(XTest)

# XTestHotEnc.head()
testPrediction = model.predict(XTestHotEnc)


# make submission





submission = pd.DataFrame({'PassengerId':testData.PassengerId,'Survived':testPrediction})

submission.head()
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)