import pandas as pd

import numpy as np

#Using to split data randomly

from sklearn.model_selection import train_test_split

#Using to get most common element from list

from collections import Counter
def readData():

    data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

    #Drop Id as it has no influence on diagnosis, and Unnamed for Nan Vals

    data = data.drop(['id', 'Unnamed: 32'], axis = 1)

    data = data.dropna()

    return data



def splitData(data, testSize):

    y = data['diagnosis'].to_numpy()

    data = data.drop(['diagnosis'], axis = 1)

    X = data.to_numpy()

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=testSize)

    return Xtrain, Xtest, ytrain, ytest
def euclidean(Xtrain, Xtestval):

    distances = []

    for i in range(len(Xtrain)):

        #euclidean equation

        distance = np.sqrt(np.sum(np.square(Xtestval-Xtrain[i])))

        distances.append([distance, i])

    return sorted(distances)
def predict(Xtrain, ytrain, Xtestval, k):

    distances = euclidean(Xtrain, Xtestval)

    predict = []

    for i in range(k):

        predict.append(ytrain[distances[i][1]])

        

    return Counter(predict).most_common(1)[0][0]
def KNN(Xtrain, Xtest, ytrain):

    predictions = []

    for i in range(len(Xtest)):

        predictions.append(predict(Xtrain, ytrain, Xtest[i], 3))

        

    return predictions
def accuracy(ytest, predictions):

    correct = 0

    for i in range(len(predictions)):

        if(predictions[i]==ytest[i]):

            correct += 1

        

    score = (correct/len(ytest))*100

    return score
data = readData()

Xtrain, Xtest, ytrain, ytest = splitData(data, 0.2)

predictions = KNN(Xtrain, Xtest, ytrain)



print(accuracy(ytest, predictions))