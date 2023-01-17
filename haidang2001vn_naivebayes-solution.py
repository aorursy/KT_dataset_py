import pandas as pd

import csv

import random

import math
!ls ../input

path = "../input/pimaindian/pima-indians-diabetes.data.csv"
def loadCsv(path):

    lines = csv.reader(open(path))

    dataset = list(lines)

    for i in range(len(dataset)):

        dataset[i] = [float(x) for x in dataset[i]]

    return dataset
dataset = loadCsv(path)
def splitDataset(dataset, splitRatio):

    trainSize = int(len(dataset) * splitRatio)

    trainSet = []

    copy = list(dataset)

    while len(trainSet) < trainSize:

        index = random.randrange(len(copy))

        trainSet.append(copy.pop(index))

    return [trainSet, copy]
dataset = [[1], [2], [3], [4], [5]]

splitRatio = 0.67 # (2/1)

train, test = splitDataset(dataset, splitRatio)

print('Split {0} rows into train with {1} and test with {2}'.format(len(dataset), train, test))
# Label is the las value in the vector

# Separated by label

def separateByClass(dataset):

    separated = {}

    for i in range(len(dataset)):

        vector = dataset[i]

        if (vector[-1] not in separated):

            separated[vector[-1]] = []

        separated[vector[-1]].append(vector)

    return separated
dataset = [[1,20,1], [2,21,0], [3,22,1]]

separated = separateByClass(dataset)

print('Separated instances: {0}'.format(separated))
def mean(numbers):

    return sum(numbers)/float(len(numbers))



def stdev(numbers):

    avg = mean(numbers)

    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)

    return math.sqrt(variance)
def summarize(dataset):

    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]

    del summaries[-1]

    return summaries
def summarizeByClass(dataset):

    separated = separateByClass(dataset)

    summaries = {}

    for classValue, instances in separated.items():

        summaries[classValue] = summarize(instances)

    return summaries
## Print out (mean, std) value for each class 

dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]

summary = summarizeByClass(dataset)

print('Summary by class value: (class:\n{0}'.format(summary))
for a in zip(*dataset):

    print(a)
## Gaussian Probability Density Function

# --- #

# We can use a Gaussian function to estimate the probability of a given attribute value, 

# given the known mean and standard deviation for the attribute estimated from the training data

# https://en.wikipedia.org/wiki/Gaussian_function

def calculateProbability(x, mean, stdev):

    exponent = math.exp( ( - ((x - mean) ** 2 ) / ( 2 * (stdev ** 2) ) ) )

    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
# Test the above function

x = 71.5

mea = 73

stde = 6.2

probability = calculateProbability(x, mea, stde)

print('Probability of belonging to this class: {0}'.format(probability))
## Calculate class probability

# --- #

def calculateClassProbabilities(summaries, inputVector):

    probabilities = {}

    for classValue, classSummaries in summaries.items():

        probabilities[classValue] = 1

        for i in range(len(classSummaries)):

            mean, stdev = classSummaries[i]

            x = inputVector[i]

            probabilities[classValue] *= calculateProbability(x, mean, stdev)

    return probabilities
# Test the above function

summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}

inputVector = [1.1, "?"]

probabilities = calculateClassProbabilities(summaries, inputVector)

print("Probabilities for each class: {0}".format(probabilities))
print(probabilities)
## Make the prediction

def predict(summaries, inputVector):

    probabilities = calculateClassProbabilities(summaries, inputVector)

    bestLabel, bestProb = None, -1

    for classValue, probability in probabilities.items():

        if bestLabel is None or probability > bestProb:

            bestProb = probability

            bestLabel = classValue

    return bestLabel
# Test the above function

summaries = {"A":[(1, 0.5)], "B":[(20, 5.0)]}

inputVector = [1.1, "?"]

result = predict(summaries, inputVector)

print("Prediction: {0}".format(result))
## Run the preduction for the set

def getPredictions(summaries, testSet):

    predictions = []

    for i in range(len(testSet)):

        result = predict(summaries, testSet[i])

        predictions.append(result)

    return predictions
summaries = {"A":[(1, 0.5)], "B":[(20, 5.0)]}

testSet = [[1.1, "?"], [19.1, "?"]]

predictions = getPredictions(summaries, testSet)

print("Predictions: {0}".format(predictions))
## Check the performance

def getAccuracy(testSet, predictions):

    correct = 0

    for x in range(len(testSet)):

        if testSet[x][-1] == predictions[x]:

            correct += 1

    return (correct/float(len(testSet))) * 100.0
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]

predictions = ['a', 'a', 'a']

accuracy = getAccuracy(testSet, predictions)

print('Accuracy: {0}'.format(accuracy))
def main():

    filename = "../input/pimaindian/pima-indians-diabetes.data.csv"

    splitRatio = 0.67

    dataset = loadCsv(filename)

    trainingSet, testSet = splitDataset(dataset, splitRatio)

    print("Split {0} rows into train={1} and test={2} rows"

          .format(len(dataset), len(trainingSet), len(testSet)))

    # prepare model

    summaries = summarizeByClass(trainingSet)

    # test model

    predictions = getPredictions(summaries, testSet)

    accuracy = getAccuracy(testSet, predictions)

    print("Accuracy: {0}%".format(accuracy))
main()