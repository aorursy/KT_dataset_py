# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

"""

    @script_author: Sasi Kiran 

    @script_name: Predict flower species using KNN Algorithm.

    @script_description: The algorithm predicts the species of flower for the given test data.

    @script_packages_used: sklearn

"""

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.datasets import load_iris

#iris = load_iris()

#print(iris)

import operator

def euclidianDistance(data1, data2, length):

    distance = 0

    for x in range(length):

        distance += np.square(data1[x] - data2[x])

       

    return np.sqrt(distance)

def knn(trainingSet, testInstance, k):

 

    distances = {}

    sort = {}

    length = testInstance.shape[1]

    print(length)

    

    

    # Calculating euclidean distance between each row of training data and test data

    for x in range(len(trainingSet)):

        dist = euclidianDistance(testInstance, trainingSet.iloc[x], length)

        distances[x] = dist[0]

       

 

    

    # Sorting them on the basis of distance

    sorted_d = sorted(distances.items(), key=operator.itemgetter(1)) #by using it we store indices also

    sorted_d1 = sorted(distances.items())

    print(sorted_d[:5])

    print(sorted_d1[:5])

   

 

    neighbors = []

    

    

    # Extracting top k neighbors

    for x in range(k):

        neighbors.append(sorted_d[x][0])

        counts = {"Iris-setosa":0,"Iris-versicolor":0,"Iris-virginica":0}

    

    

    # Calculating the most freq class in the neighbors

    for x in range(len(neighbors)):

        response = trainingSet.iloc[neighbors[x]][-1]

 

        if response in counts:

            counts[response] += 1

        else:

            counts[response] = 1

  

    print(counts)

    sortedVotes = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

    print(sortedVotes)

    return(sortedVotes[0][0], neighbors)



testSet = [[1.8, 1.6, 3.4, 1.2]]

test = pd.DataFrame(testSet)

iris = pd.read_csv("../input/iris/Iris.csv")

result,neigh = knn(iris, test, 4)#here we gave k=4

print("And the flower is:",result)

print("the neighbors are:",neigh)# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.