%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pylab import rcParams



from scipy import stats

from sklearn.metrics import mean_squared_error as MSE



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.linear_model import Lasso
trainingData = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

competitionData = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print(trainingData.shape)

print(competitionData.shape)
display(trainingData.head())

display(competitionData.head())
TrainingDataColumns = list(trainingData)
for c in TrainingDataColumns:

    if trainingData[c].dtype == 'O':

        trainingData[c].fillna(value = 'Missing', inplace = True)

    else:

        trainingData[c].fillna(0, inplace = True)
competitionDataColumns = list(competitionData)
for f in competitionDataColumns:

    if competitionData[f].dtype == 'O':

        competitionData[f].fillna(value = 'Missing', inplace = True)

    else:

        competitionData[f].fillna(0, inplace = True)
display(trainingData.head())

display(competitionData.head())
trainingData = pd.get_dummies(trainingData)

competitionData = pd.get_dummies(competitionData)
display(trainingData.head())

display(competitionData.head())
print(trainingData.shape)

print(competitionData.shape)
sp = trainingData['SalePrice']

missingFeatures = list(set(trainingData.columns.values) - set(competitionData.columns.values))

trainingData = trainingData.drop(missingFeatures, axis=1)
missingFeatures = list(set(competitionData.columns.values) - set(trainingData.columns.values))

competitionData = competitionData.drop(missingFeatures, axis=1)
print(trainingData.shape)

print(competitionData.shape)
X_train, X_test, y_train, y_test = train_test_split(trainingData, sp, random_state=0)
myModel = Lasso(alpha=298.4).fit(X_train,y_train)

print("Train Score: ", myModel.score(X_train, y_train), "\n Test Score: ", myModel.score(X_test, y_test))

print("Number of Features user: {}".format(np.sum(myModel.coef_ != 0)))
submission = pd.DataFrame(myModel.predict(competitionData), columns=['SalePrice'], index = competitionData['Id'])

display(submission.head())
submission.to_csv("submission_joelpires.csv")
# DISCLAIMER: THIS CODE IS A RESULT OF FOLLOWING THIS TUTORIAL: youtube.com/watch?v=3S9j71OL1H0