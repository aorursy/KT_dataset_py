import pandas as pd

import numpy as np



# Read the data from the CSV

data = pd.read_csv("../input/diabetes.csv", header=0).as_matrix()
# Split up the matrix into labels and feature vectors

featureVectors = data[:,:-1]

labels = data[:,-1:]



# Take a certain percentage of the data as training data and the rest for

# evaluating the performance

trainingDataRatio = 0.7

numberOfExamples = data.shape[0]



print("Number of training examples over all:", numberOfExamples)



trainingData = np.array([[]])

trainingLabels = np.array([[]])

testData = np.array([[]])

testLabels = np.array([[]])



for i in range(0, data.shape[0]):

    if(np.random.uniform(0, 1)>trainingDataRatio):

        testData = np.append(testData, featureVectors[i])

        testLabels = np.append(testLabels, labels[i])

    else:

        trainingData = np.append(trainingData, featureVectors[i])

        trainingLabels = np.append(trainingLabels, labels[i])



numberOfTrainingExamples = int(trainingData.shape[0]/8)

numberOfTestExamples = numberOfTrainingExamples - numberOfExamples



trainingData = np.reshape(trainingData, (numberOfTrainingExamples, 8))

testData = np.reshape(testData, (numberOfTestExamples, 8))



print("Training Data: #", trainingData.shape)

print("Test Data: #", testData.shape)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=30, criterion='gini')



clf = clf.fit(trainingData, trainingLabels, sample_weight=None)



meanAccuracy = clf.score(testData, testLabels)



print("Mean Accuracy", meanAccuracy)