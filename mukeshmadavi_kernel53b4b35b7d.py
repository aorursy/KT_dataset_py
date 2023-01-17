# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input"))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import math, random,operator 

def euclideanDistance(instance1, instance2, length):

	distance = 0

    

	for x in range(length):

		distance+= pow((instance1[x] - instance2[x]), 2)

	return float(math.sqrt(float(distance)))
def getNeighbors(trainingSet, testInstance, k):

	distances = []

	length = len(testInstance)-1

   

	for x in range(len(trainingSet)):

		dist = euclideanDistance(testInstance, trainingSet[x], length)

		distances.append((trainingSet[x], dist))

	distances.sort(key=operator.itemgetter(1))

	neighbors = []

	for x in range(k):

		neighbors.append(distances[x][0])

	return neighbors
def getResponse(neighbors):

	classVotes = {}

	for x in range(len(neighbors)):

		response = neighbors[x][-1]

		if response in classVotes:

			classVotes[response] += 1

		else:

			classVotes[response] = 1

	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), 

                      reverse=True)

	return sortedVotes[0][0]
def getAccuracy(testSet, predictions):

    correct = 0

    for x in range(testsize):

        print(testSet[x][-1], ".........", predictions[x] )

        if testSet[x][-1] == predictions[x]:

            correct += 1

    return (correct/testsize) * 100.0



#How many do we test?

testsize = random.randint(10,400)
def main():

	# prepare data

 

   #dataset = pd.read_csv("/kaggle/input/braincsv/bt_dataset_t3.csv") 

    a = pd.read_csv("/kaggle/input/medium.csv",sep=",")

    a = a.drop(a.columns[0], axis=1)

    #a = a.drop(a.columns[4:], axis=1)

    print("read the dataset")

    

    tr = a.values[:1000]

    test = a.values[1001:]

     # print(tr)

   # print(test)

    print("train test split")

	# generate predictions

    predictions=[]

    k = 5

    print("test size",testsize)

    for x in range(testsize):

        print("inside for loop")

        neighbors = getNeighbors(tr, test[x], k)

        print("returned neighbors")

        result = getResponse(neighbors)

       # print("retuned result",result)

        predictions.append(result)

        print("appended")

       # print(predictions)

    print("outside for loop.....")

    accuracy = getAccuracy(test, predictions)

    

    print('Accuracy for small dataset ', accuracy, '%')

main()