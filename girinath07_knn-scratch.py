# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import math

import operator

import random

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/iris.csv")

#x = data.iloc[:,0:4]

#y = data.iloc[:,4:5]

#print("first five rows.....")

#print(x.head())

#print("last five rows...")

#print(x.tail()) 

#print(x.iloc[1])

#print(data.iloc[0,-1]=='setosa')

for row in range(0,150):

    if data.iloc[row,-1]  == 'setosa' :

        data.iloc[row,-1]=0

    elif data.iloc[row,-1]  == 'versicolor' :

        data.iloc[row,-1]=1

    elif data.iloc[row,-1]  == 'virginica' :

        data.iloc[row,-1]=2
trainingSet=[]

testSet=[]

split=0.66

random.seed(30)

for r in range(len(data)):

    if random.random() < split:

        trainingSet.append(list(data.values[r]))

    else:

        testSet.append(list(data.values[r]))

#print(trainingSet)
def ED(x1, x2, length): #it is used for calculating euclidean distance

    distance = 0

    for x in range(length):

        distance += np.square(x1[x] - x2[x])

        #print(f"x={x} || x1[x]={x1[x]} || x2[x]={x2[x]} || dist={distance}")

    return np.sqrt(distance)
def getNeighbors(trainingSet, testInstance, k):

    distances = []

    length = len(testInstance)-1

    for x in range(len(trainingSet)):

        dist = ED(testInstance, trainingSet[x], length)

        distances.append((trainingSet[x], dist))

        #print(f"dis={distances}")

    distances.sort(key=operator.itemgetter(1))

    #print(f"dis={distances}")

    neighbors = []

    for x in range(k):

        neighbors.append(distances[x][0])

        #print(f"x={x} || distances={distances[x][0]} || neighbors={neighbors}")

    return neighbors
def getResponse(neighbors):

    classVotes = {}

    for x in range(len(neighbors)):

        response = neighbors[x][-1]

        #print(f"res={response} || nei={neighbors}")

        if response in classVotes:

            classVotes[response] += 1

            #print(f"if_classsVotes={classVotes}")

        else:

            classVotes[response] = 1

            #print(f"else_classsVotes={classVotes}")

    #print(f"classVotes={classVotes}")

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    #print(f"sortedVotes={sortedVotes} || sortedVotes[0][0]={sortedVotes[0][0]}")

    return sortedVotes[0][0]
def getAccuracy(testSet, predictions):

    correct = 0

    for x in range(len(testSet)):

        #print(testSet[x][-1])

        if (testSet[x][-1] - predictions[x]) == 0.0:

            correct += 1

    return (correct/float(len(testSet))) * 100.0
def knn():

    # generate predictions

    predictions=[]

    k = 3

    #print(trainingSet)

    for x in range(len(testSet)):

        #print(f"testSet[x]={testSet[x]} || k={k}")

        neighbors = getNeighbors(trainingSet, testSet[x], k)

        result = getResponse(neighbors)

        predictions.append(result)

        #print(f"predictions={predictions}")

        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

    accuracy = getAccuracy(testSet, predictions)

    print('Accuracy: ' + repr(accuracy) + '%')
knn()