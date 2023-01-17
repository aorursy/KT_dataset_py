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
import csv
import random
import math
import operator


# Loads the data set and then call make float to convert all data into floats so that nothing is a string
def loadDataset(trainingSet=[], testSet=[]):
    with open("../input/smallDataSet.csv", 'r') as csvfile:
        # reads through the rows in the csv and then returns them as strings
        lines = csv.reader(csvfile)
        # creates a list from the rows in the dataset
        dataset = list(lines)
        # since csv.reader returns everything as string, this makes a call that converts them all to float
        dataset = makeFloat(dataset)
        # splits the whole dataset into training data and test data using the random function to roughly split it 66 33
        for x in range(len(dataset) - 1):
            if random.random() < .67:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


# takes in all of the rows from the dataset and then converts them from string to float by casting type on it
def makeFloat(ds):
    for x in range(len(ds) - 1):
        for y in range(6):
            ds[x][y] = float(ds[x][y])
    return ds


# calculates the euclidian distance between two points
def findEuclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# compares the row from the test data to values in training data so that it can find the nearest neighbors
def get5NearestNeighbors(trainingSet, testInstance):

    distances = []
    # to avoid arrayIndexOutOfBounds Exception
    length = len(testInstance) - 1
    # loop to run through the trainingSet and find the nearest neighbors of each instance
    for x in range(len(trainingSet)):
        dist = findEuclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    # sorts the distances
    distances.sort(key=operator.itemgetter(1))
    fiveNeighbors = []
    # adds the five closest to the list of 5 neighbors
    for x in range(5):
        fiveNeighbors.append(distances[x][0])
    return fiveNeighbors


# gets the guess for the machiene to use based on nearest neighbors
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# calculates the accuracy of the nearest neighbors guesses
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():

    # define trainingSet and testSet as two empty arrays
    trainingSet = []
    testSet = []
    # load the dataset into the two sets so that k nearest neighbor can be used
    loadDataset(trainingSet, testSet)

    # prints the size of each the training and test set to see how random split up the data
    print('Train set: ' + str(len(trainingSet)))
    print('Test set: ' + str(len(testSet)))

    # generate predictions
    predictions = []

    # runs through the testSet of data and gets the nearest neighbors and prints the results
    for x in range(len(testSet)):
        # calls get5NearestNeighbors and then gets the nearest ne
        neighbors = get5NearestNeighbors(trainingSet, testSet[x])
        # Determines if the project would be successful or not
        result = getResponse(neighbors)
        predictions.append(result)
        # prints the predicted result compared to the actual
        print('predicted=' + str(result) + ', actual=' + str(testSet[x][-1]))
    # Calculates the accuracy of the learner
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + str(accuracy) + '%')


main()
