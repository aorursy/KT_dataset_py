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
import pandas as pd

import numpy as np

import math

import operator



data = pd.read_csv("../input/iris.csv")
data.head()
def euclidianDistance(data1,data2,length):

    distance = 0

    for x in range(length):

        distance += np.square(data1[x]-data2[x])

    return np.sqrt(distance)



def knn(trainingSet,testInstance,k):

    distances ={}

    sort ={}

    length = testInstance.shape[1]

    for x in range(len(trainingSet)):

        dist = euclidianDistance(testInstance,trainingSet.iloc[x],length)

        distances[x] = dist[0]

    

    

    sorted_d = sorted(distances.items(),key=operator.itemgetter(1))

    neighbors =[]

    

    for x in range(k):

        neighbors.append(sorted_d[x][0])

        

    classVotes = {}

    

    

    for x in range(len(neighbors)):

        response = trainingSet.iloc[neighbors[x]][-1]

        if response in classVotes:

            classVotes[response] += 1

        else:

            classVotes[response] = 1

            

    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)

    return(sortedVotes[0][0],neighbors)



        

        

    

    

    
testSet = [[7.2, 3.6, 5.1, 2.5]]

test = pd.DataFrame(testSet)


k = 5

result,neigh = knn(data, test, k)

print(result)
print(neigh)
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(data.iloc[:,0:4], data['Species'])



# Predicted class

print(neigh.predict(test))



# 3 nearest neighbors

print(neigh.kneighbors(test)[1])
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



data = data.apply(LabelEncoder().fit_transform)

y = data['Species']



X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(n_estimators=10, max_depth=2,random_state=0)

clf.fit(X_train, y_train)
y_testpred= clf.predict(X_test)
y_testpred
X_test