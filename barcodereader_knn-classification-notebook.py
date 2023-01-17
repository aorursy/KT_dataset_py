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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import random
from collections import Counter
import matplotlib.pyplot as plt
#====================================Functions to perform KNN====================================
#Function 1: Import Data
#input: file path, and total column of data(including label column)
#output: data matrix (line by line), label matrix, column of data(minus last column)
#this function provides 2 modes of import, either offline file or pandas
def importdata(filepath,numberOfCol,mode = 'offline'):
    if mode == 'offline':
        file = open(filepath,'r') #read the file
        numberOfLines = len(file.readlines()) #calculate how many lines are in the txt file
        returnMat = np.zeros([numberOfLines,numberOfCol-1]) #create a 0 matrix to contain each line. this is a numpy array.the reason to -1 is to ignore the title line(first line)
        classLabelVector = [] #empty list for overall data
        index = 0
        file = open(filepath,'r') # to let the .readlines() function read from begining again
        for line in file.readlines(): #readlines will return a list of lines
            line = line.strip() #to get each seperate line
            listFromLine = line.split(',') #to cut each line into individual strings
            returnMat[index,:] = listFromLine[0:numberOfCol-1] 
            classLabelVector.append(int(listFromLine[-1])) #this is the vector to store the label, -1 means read from right to left
            index += 1
    if mode == 'pandas':
        file = pd.read_csv(filepath)
        lines = len(file['TARGET CLASS'])
        nameTag = ('XVPM','GWYH','TRAT','TLLZ','IGGA','HYKR','EDFS','GUUB','MGJM','JHZC')
        returnMat = list()
        classLabelVector = list()
        for i in range(1,lines): #for each line
            eachline = list()
            for j in range(0,numberOfCol-1): #for each column, exclude the last one
                eachline.append(file[nameTag[j]][i])
            returnMat.append(eachline)
            classLabelVector.append(file['TARGET CLASS'][i])
    return returnMat, classLabelVector, numberOfCol-1
#Function 2: Normalize
#this is a function trying to eleminate the effect of large numbers. For example, distance for 1000 to 1600 has a larger impact than 1 to 10. Thus we normalize the data to (0,1)
#input: raw data martix, column number of data(minus label)
#output: normalized data matrix
def normalize(dataMat,numberOfCol):
    lendataMat = len(dataMat)
    for i in range(lendataMat):
        maxi = max(dataMat[i])
        mini = min(dataMat[i])
        for j in range(numberOfCol):
            dataMat[i][j] = (dataMat[i][j] - mini)/(maxi - mini)
    return dataMat
#Function 3: Divide the data into training set and testing set
#trimed data matrix, label matrix, testing rate
#output: training set, testing set, trainlabel set, test label set. based on the ratio defined
#important: the actual number of elements in testing set will be lesser, this is because when generating random numbers, some of them are the same, thus the unique index in less than our expectation
def trainNtest(dataMat,labelMat, testRate):
    lendataMat = len(dataMat)
    testVolume = math.floor(lendataMat*testRate) #the largest integer less than lendataMat*testRate
    seed = set() #index number
    for i in range(testVolume):
        seed.add(random.randint(0,lendataMat-1)) # -1 is because the end point is included
    index = [i for i in range(lendataMat)]
    remainder = np.delete(index, list(seed))
    testSet = [dataMat[i] for i in seed]
    testLabel = [labelMat[i] for i in seed]
    trainSet = [dataMat[i] for i in remainder]
    trainLabel = [labelMat[i] for i in remainder]
    return trainSet, testSet, trainLabel, testLabel
#Function 4: KNN itself
#input: value K, training set, testing set, training classification, column of attributes, and running mode
#output: the prediction / classification
#important: running mode has 3 types: 
    #hist: this is the standard KNN, judge the result by highest probability (presence)
    #dist: this is the distance based judgement, ignore probability, just based on closest distance (don't care about the presence)
    #mix: this is the combination of above 2, when the chance is 50-50, use the distance to judge, when the chance is not 50-50, use probability
def KNN(K, trainSet, testSet,trainLabel, col, stdmode = 'hist'):
    lentrainSet = len(trainSet)
    lentestSet = len(testSet)
    predict = list() #for label prediction
    label = set(trainLabel) #this will give type of labels
    hist = dict()
    for i in range(lentestSet):
        result = list() #for distance
        for j in range(lentrainSet):
            suma = 0
            for k in range(col):
                suma += (testSet[i][k] - trainSet[j][k])**2
            result.append(math.sqrt(suma)) #the distance
        sortarray = np.argsort(result) # index number in ascending arrangement
        karray = [trainLabel[i] for i in sortarray[:K]] #the first K elements in label matrix
        darray = [result[i] for i in sortarray[:K]] #the first K elements in distance matrix
        #print(darray)
        hist = {i:karray.count(i) for i in label} #label is a set with all classification inside
        if stdmode == 'hist': #use the highest portion, i.e. the probability to predict the classification
            test = max([hist[i] for i in hist]) 
            final = [i for i in hist if hist[i] == test][0] #if the probability is 50-50, always use the first one.
            predict.append(final)
        if stdmode == 'dist': #use the mean average distance to determine the classification
            matchTable = {karray[i]:[] for i in range(K)}
            for i in range(K):
                matchTable[karray[i]].append(darray[i])
            mean = {karray[i]:(sum(matchTable[karray[i]])/len(matchTable[karray[i]])) for i in label}
            minimum = min([mean[i] for i in mean])
            result = [i for i in mean if mean[i] == minimum][0]
            predict.append(result)
        if stdmode == 'mix':
            test = max([hist[i] for i in hist]) 
            if test == K/2: #if the probability is 50-50
                matchTable = {karray[i]:[] for i in range(K)}
                for i in range(K):
                    matchTable[karray[i]].append(darray[i])
                mean = {karray[i]:(sum(matchTable[karray[i]])/len(matchTable[karray[i]])) for i in label}
                minimum = min([mean[i] for i in mean])
                result = [i for i in mean if mean[i] == minimum][0]
                predict.append(result)
            else:
                final = [i for i in hist if hist[i] == test][0]
                predict.append(final)
    return predict
#Function 5: Error Rate
#input: predicted list, true value
#output: a dictionary of {True value : Predicted Value}, and an error rate
def errRate(predict, test):
    if len(predict) != len(test):
        print("2 sets dimension not the same: " + str(len(predict)) + " vs " + str(len(test)))
    else:
        wrong = 0
        table = [{predict[i] : test[i]} for i in range(len(test))]
        for i in range(len(test)):
            if predict[i] != test[i]:
                wrong+=1
        #print("True value : Predicted Value")
        #print(table)
        err = wrong/len(test)
        #print("Error Rate: " + str(wrong/len(test)))
        return err
#====================================Functions Ends Here====================================
#====================================Body of KNN Analysis====================================
#import data
dataMat, labelMat, col = importdata('../input/KNN_Project_Data',11,mode = 'pandas')
#nomalize data
dataMat = normalize(dataMat,col)
#random picking test data
testRate = 0.3 #use 30% of the data to test model
trainSet, testSet, trainLabel, testLabel = trainNtest(dataMat,labelMat, testRate)
#calculate result and find K with smallest error

error = list()
errdic = dict()
for i in range(1,100):
    err= 0
    predictLabel = KNN(i, trainSet, testSet,trainLabel,col, stdmode='hist') # col means the total colume number of data set(exclude label)
    #error rate
    err += errRate(predictLabel, testLabel)
    error.append(err)
    errdic[i] = "{0:.2f}".format(err)
#graph plot
plt.plot(range(1,100),error,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print(errdic)