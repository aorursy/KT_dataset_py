import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import copy as cp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from statistics import mode
import os.path
print (os.listdir("../input"))
trainData = pd.read_csv("../input/adultds/train_data.csv", header=0, index_col=0, na_values="?")
testData = pd.read_csv("../input/adultds/test_data.csv", header=0, index_col=0, na_values="?")
trainData.shape
trainData.head()
trainData["native.country"].value_counts()
trainData["age"].value_counts().plot(kind="pie")
trainData["sex"].value_counts().plot(kind="bar")
trainData["occupation"].value_counts().plot(kind="bar")
trainData["relationship"].value_counts().plot(kind="pie")
trainData = trainData.drop(columns="education") # Drop education column because education.num contains the same data
testData  = testData.drop(columns="education") # Drop education column because education.num contains the same data
allData = []
allData = pd.concat([testData, trainData], sort=False)
allData
completeTrainData = cp.copy(trainData)
completeTestData  = cp.copy(testData)

allCols  = ["age","workclass","fnlwgt","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week","native.country"]
modeCols = ["workclass","education.num","marital.status","occupation","relationship","race","sex","native.country"]
meanCols = ["age","fnlwgt","capital.gain","capital.loss","hours.per.week"]
encoders = {}
modes = {}
means = {}
fillers = {}
for col in allCols:
    encoders[col] = LabelEncoder().fit(allData[col].dropna())
for col in modeCols:
    modes[col] = mode(encoders[col].transform(trainData[col].dropna()))
    completeTrainData[col] = trainData[col].fillna(encoders[col].inverse_transform([int(modes[col])])[0])
    completeTestData[col]  = testData[col].fillna(encoders[col].inverse_transform([int(modes[col])])[0])
for col in modeCols:
    completeTrainData[col] = encoders[col].transform(completeTrainData[col])
    completeTestData[col]  = encoders[col].transform(completeTestData[col])
for col in meanCols:
    means[col] = np.mean(trainData[col].dropna())
    completeTrainData[col] = trainData[col].fillna(means[col])
    completeTrainData[col] = encoders[col].transform(completeTrainData[col])
    completeTestData[col]  = testData[col].fillna(means[col])
    completeTestData[col]  = encoders[col].transform(completeTestData[col])
    
completeTrainData
minmaxscaler = MinMaxScaler()
data = ["age","workclass","education.num","marital.status", "occupation", "relationship", "race", "capital.gain", "capital.loss", "hours.per.week", "native.country"]

trainDataX = completeTrainData[data]
trainDataX = minmaxscaler.fit_transform(trainDataX)
testDataX  = completeTestData[data]
testDataX  = minmaxscaler.transform(testDataX)
trainDataY = completeTrainData["income"]
meanScoreManhattan = np.zeros(50)
stdScoreManhattan  = np.zeros(50)
for k in range(1,51):
    classifier = KNeighborsClassifier(n_neighbors=k, p=1)
    score = cross_val_score(classifier, trainDataX, trainDataY, cv=10)
    meanScoreManhattan[k-1] = np.mean(score)
    stdScoreManhattan[k-1]  = np.std(score)
    
np.amax(meanScoreManhattan)
meanScoreEuclidean = np.zeros(50)
stdScoreEuclidean  = np.zeros(50)
for k in range(1,51):
    classifier = KNeighborsClassifier(n_neighbors=k, p=2)
    score = cross_val_score(classifier, trainDataX, trainDataY, cv=10)
    meanScoreEuclidean[k-1] = np.mean(score)
    stdScoreEuclidean[k-1]  = np.std(score)
    
np.amax(meanScoreEuclidean)
if np.amax(meanScoreManhattan) > np.amax(meanScoreEuclidean):
    chosenK = np.argmax(meanScoreManhattan)+1
    chosenP = 1
else:
    chosenK = np.argmax(meanScoreEuclidean)+1
    chosenP = 2
    
chosenK
chosenP
plt.errorbar(range(1,51), meanScoreManhattan, yerr=1.96*np.array(stdScoreManhattan), fmt='-o')
plt.errorbar(range(1,51), meanScoreEuclidean, yerr=1.96*np.array(stdScoreEuclidean), fmt='-o')
classifier = KNeighborsClassifier(n_neighbors=chosenK,p=chosenP)
classifier.fit(trainDataX,trainDataY)
predictedData = classifier.predict(testDataX)
predictedData
output = pd.DataFrame(testData.index)
output["income"] = predictedData
output
output.to_csv("PMR3508_MarcusPavani_Adult.csv", index=False)