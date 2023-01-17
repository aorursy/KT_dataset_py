# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import math

import operator



# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/Iris.csv')

iris.head()
iris.drop('Id',axis=1,inplace=True)
# Creating a function which calculates the distance (eucledian distance) between two data points

def euclidian_distance(dp1,dp2,NoOfColumns):

    distance=0

    for i in range(NoOfColumns):

        distance += np.square ( dp1[i] - dp2[i] )

    return np.sqrt(distance)



# Testing the function 



dp1 = [1,3,4,'b']

dp2 = [3,2,3,'c']

NoOfColumns = 3 

testdistance = euclidian_distance(dp1,dp2,NoOfColumns)

print(testdistance)
nb = [[1,2,3,4],

     [5,6,7,8]]

nb[0][-1]
# Creating / Defining the KNN model 



def knn(TrainingData, testData, k) :

    distances ={}

    sort = {}

    NoOfColumns = testData.shape[1]

    

    # Calculating euclidean distance between each row of training data and test data

    for i in range(len(TrainingData)):

        InterimDistance = euclidian_distance(TrainingData.iloc[i], testData, NoOfColumns)

        distances[i] = InterimDistance[0]

    

    # Sorting them on the basis of distance

    SortedDistance = sorted(distances.items(),key=operator.itemgetter(1))

    # Identifying and extracting the top K neighbours 

    neighbour = []

    for i in range(k):

        neighbour.append(SortedDistance[i][0])

        

    # Calculating the most freq class/group in the neighbors

    classvotes = {}

    for i in range(len(neighbour)):

        #ClassigiedAs = TrainingData.iloc[][-1] # TrainingData is 2 dimensional array returning value of last column (category of Iris)

        ClassifiedAs = TrainingData.iloc[neighbour[i]][-1]

        if ClassifiedAs in classvotes :

            classvotes[ClassifiedAs] += 1

        else :

            classvotes[ClassifiedAs] = 1

    SortedVotes = sorted(classvotes.items(), key=operator.itemgetter(1), reverse=True)

    return(SortedVotes[0][0],neighbour)

    
testing = pd.DataFrame([[7.2, 3.6, 5.1, 2.5]])

kvalue = 5

#testing

result,nb = knn(iris,testing,kvalue)
print("Classified as :", result)

print(nb)
from sklearn.neighbors import KNeighborsClassifier
nbb = KNeighborsClassifier(n_neighbors = 5 )

nbb.fit(iris.iloc[:,0:4], iris['Species'])

print(nbb.predict(testing))

print(nbb.kneighbors(testing)[1][0])