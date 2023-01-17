import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import GaussianNB # Naive Bayes classifier

import matplotlib.pyplot as plt

%matplotlib inline



dfTrain = pd.read_csv("../input/train.csv") # importing training set

dfTest = pd.read_csv("../input/test.csv") # importing test set
dfTrain.shape
print(dfTrain.isnull().sum())
print(dfTest.isnull().sum())
dfTrain.fillna(dfTrain.mean(), inplace=True)

dfTest.fillna(dfTest.mean(), inplace=True)

ax = dfTrain.Survived.value_counts().plot(kind='bar',color="rg")

ax.set_ylabel("Number of passengers")

ax.set_xlabel("Survived [No-0 , Yes -1]")

rects = ax.patches



# Add labels



for rect in rects:

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, 1 + height, '%d' % int(height), ha='center', va='bottom')

dfTrain['Sex'] = pd.factorize(dfTrain['Sex'])[0]

dfTest['Sex'] = pd.factorize(dfTest['Sex'])[0]
trainAttributeData = pd.DataFrame.as_matrix(dfTrain[['Pclass','Age','Sex','Fare']])

testAttributeData =  pd.DataFrame.as_matrix(dfTest[['Pclass','Age','Sex','Fare']])

trainPredictAttribute =  pd.DataFrame.as_matrix(dfTrain[['Survived']]).ravel()
trainPredictAttribute.shape

classifier = GaussianNB()

classifier.fit(trainAttributeData,trainPredictAttribute)

GaussianNB(priors=None)
predictValues = pd.DataFrame(classifier.predict(testAttributeData),columns=['Survived'])
passengerIdValues = pd.DataFrame()

passengerIdValues['PassengerId'] = dfTest['PassengerId']
finalResult = passengerIdValues.join(predictValues)
finalResult.to_csv("Survivor_prediction.csv", index = False)
finalResult.head()