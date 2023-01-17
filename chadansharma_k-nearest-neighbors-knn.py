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
# Importing libraries

import pandas as pd

import numpy as np

import math

import operator
data = pd.read_csv("../input/IRIS.csv")
# Defining a function which calculates euclidean distance between two data points

def euclideanDistance(data1, data2, length):

    distance = 0

    for x in range(length):

        distance += np.square(data1[x] - data2[x])

    return np.sqrt(distance)



# Defining our KNN model

# This is the approach a Brute force, where distance from all values are calculated then nearest neighbours 

# with least distance are identified

# Brute force performs worst when there are large dimensions and large training set

def knn(trainingSet, testInstance, k):

 

    distances = {}

    sort = {}

 

    length = testInstance.shape[1]

    

    #### Start of STEP 3

    # Calculating euclidean distance between each row of training data and test data

    for x in range(len(trainingSet)):

        

        #### Start of STEP 3.1

        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist[0]

        #### End of STEP 3.1

 

    #### Start of STEP 3.2

    # Sorting them on the basis of distance

    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    #### End of STEP 3.2

 

    neighbors = []

    

    #### Start of STEP 3.3

    # Extracting top k neighbors

    for x in range(k):

        neighbors.append(sorted_d[x][0])

    #### End of STEP 3.3

    

    classVotes = {}

    

    #### Start of STEP 3.4

    # Calculating the most freq class in the neighbors

    for x in range(len(neighbors)):

        response = trainingSet.iloc[neighbors[x]][-1]

 

        if response in classVotes:

            classVotes[response] += 1

        else:

            classVotes[response] = 1

#         print (classVotes)

    #### End of STEP 3.4



    #### Start of STEP 3.5

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    print (sortedVotes)

    return(sortedVotes[0][0], neighbors)

    #### End of STEP 3.5

testSet = [[7.2, 3.6, 5.1, 2.5]]

test = pd.DataFrame(testSet)
# Setting number of neighbors = 1

k = 1

#### End of STEP 2

# Running KNN model

result,neigh = knn(data, test, k)



# Predicted class

print(result)

print(neigh)
# Setting number of neighbors = 3 

k = 3 

# Running KNN model 

result,neigh = knn(data, test, k) 

# Predicted class 

print(result)
k = 5

# Running KNN model 

result,neigh = knn(data, test, k) 

# Predicted class 

print(result)

print(neigh)
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree',leaf_size=5)

neigh.fit(data.iloc[:,0:4], data['species'])



# Predicted class

print(neigh.predict(test))



# 3 nearest neighbors

print(neigh.kneighbors(test)[1])
