# -*- coding: utf-8 -*-

import pandas as pd 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Open data csvs

titanicTrain = pd.read_csv('/kaggle/input/titanic/cleanTitanicTrain.csv')

titanicTest = pd.read_csv('/kaggle/input/titanic/cleanTitanicTest.csv')



# Columns used to predict

predictorNames = ["Pclass", 

                  "Sex", 

                  "Age", 

                  "SibSp", 

                  "Parch", 

                  "Fare", 

                  "Embarked"]



# Create the predictors and targets

trainPredictors = titanicTrain[predictorNames]

targetName = ["Survived"]

trainTarget = titanicTrain["Survived"]

testPredictors = titanicTest[predictorNames]

testPIDs = titanicTest['PassengerId']



# Randomly sample the train set to match the length of the test set

trainPredictors = trainPredictors.sample(len(testPredictors))

trainTarget = trainTarget.sample(len(testPredictors))



# Create the random forest classifier

RandomForest = RandomForestClassifier(n_estimators = 200,

                                      criterion = 'gini',

                                      min_samples_split = 5,

                                      min_samples_leaf = 5,

                                      max_features = 5)

RandomForest.fit(trainPredictors,

                 trainTarget)

RFPredictions = RandomForest.predict(testPredictors)



# Print the confusion matric and classification report

print(confusion_matrix(trainTarget,

                       RFPredictions))

print(classification_report(trainTarget,

                            RFPredictions))

predictedTarget = RFPredictions



# Create the submission csv

submission = pd.DataFrame({'PassengerID': testPIDs,

                           'Survived': predictedTarget})

submission.to_csv('submission.csv',

                  index = False)
