# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import csv

# Any results you write to the current directory are saved as output.
def toInt(array):
    array=np.mat(array)
    m,n=array.shape
    newArray=np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            newArray[i,j]=int(array[i,j])
    return newArray

def tooInt(array):
    array=np.mat(array).T
    m,n=array.shape
    newArray=np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            newArray[i,j]=int(array[i,j])
    return newArray

def nomalizing(array):
    array=np.mat(array)
    m,n=array.shape
    for i in range(m):
        for j in range(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array
def loadTrainData():
    l=[]
    with open('../input/train.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l=np.array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),nomalizing(tooInt(label))
    
    
    
b,c=loadTrainData()
print(b.shape)
print(c.shape)
def loadTestResult():
    l=[]
    with open('../input/sample_submission.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    
    label=np.array(l)
    return tooInt(label[:,1])
t=loadTestResult()
print(t.shape)
def loadTestData():
    l=[]
    with open ('../input/test.csv') as myfile:
        lines=csv.reader(myfile)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    
    data=np.array(l)
    return nomalizing(toInt(data))
    
u=loadTestData()
print(u.shape)
def classify(inX,dataSet,labels,k):
    
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=np.array(diffMat)**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        voteIlabel=np.array(voteIlabel)
        classCount[voteIlabel[0][0]]=classCount.get(voteIlabel[0][0],0)+1
        
        
    result = sorted(classCount.items(), reverse=True)
    return result[0][0]
    
    
    
    
    
    
    
testData=loadTestData()
trainData,trainLabel=loadTrainData()
testLabel=loadTestResult()
u=classify(testData[0],trainData,trainLabel,5)
print(u)

    

def saveResult(result):
    with open('../input/sample_submission.csv',w) as file:
        myWriter=csv.writer(file)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)
            

o=np.array((loadTestResult()))
print(o.shape)
trainData,trainLabel=loadTrainData()
testData=loadTestData()
testLabel=loadTestResult()
def handwritingClassTest(trainData,trainLabel,testData):
   
    m,n=testData.shape
    errorCount=0
    resultList=[]
    for i in range(m):
        classifierResult=classify(testData[i],trainData,trainLabel,5)
        resultList.append(classifierResult)
      
       
    saveResult(resultList)
    

    
trainData,trainLabel=loadTrainData()
testData=loadTestData()

handwritingClassTest(trainData,trainLabel,testData)
