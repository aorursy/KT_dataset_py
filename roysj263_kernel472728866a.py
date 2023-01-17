import pandas as pd
import numpy as np
import math
import operator
data = pd.read_csv('../input/iris.csv')
print(data.head(5)) 
testSet = [[2.3,2.9,5.6,0.8]]
test = pd.DataFrame(testSet)
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)
def knn(trainingSet, testInstance, k):
    distances = {}
    sort = {}
    length = testInstance.shape[1]
    for x in range(len(trainingSet)):    
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    classVotes = {}
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return(sortedVotes[0][0], neighbors)
for i in range(1,4):
    print(i)
    result,neigh = knn(data, test, i)
    print('\nPredicted Class of the datapoint = ', result)
    print('\nNearest Neighbour of the datapoints = ',neigh)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data.iloc[:,0:4], data['Name'])
print(neigh.predict(test))
print(neigh.kneighbors(test)[1])
