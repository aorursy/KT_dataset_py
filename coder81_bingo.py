from numpy import *

import csv



def toInt(array):

    array=mat(array)

    m,n=shape(array)

    newArray=zeros((m,n))

    for i in xrange(m):

        for j in xrange(n):

                newArray[i,j]=int(array[i,j])

    return newArray

    

def nomalizing(array):

    m,n=shape(array)

    for i in xrange(m):

        for j in xrange(n):

            if array[i,j]!=0:

                array[i,j]=1

    return array
def loadTrainData():

    l=[]

    with open('train.csv') as file:

         lines=csv.reader(file)

         for line in lines:

             l.append(line) #42001*785

    l.remove(l[0])

    l=array(l)

    label=l[:,0]

    data=l[:,1:]

    return nomalizing(toInt(data)),toInt(label)  

    

def loadTestData():

    l=[]

    with open('test.csv') as file:

         lines=csv.reader(file)

         for line in lines:

             l.append(line)

    l.remove(l[0])

    data=array(l)

    return nomalizing(toInt(data))  

   

    

def loadTestResult():

    l=[]

    with open('knn_benchmark.csv') as file:

         lines=csv.reader(file)

         for line in lines:

             l.append(line)#28001*2

    l.remove(l[0])

    label=array(l)

    return toInt(label[:,1])  #  label 28000*1

    



def saveResult(result,csvName):

    with open(csvName,'wb') as myFile:    

        myWriter=csv.writer(myFile)

        for i in result:

            tmp=[]

            tmp.append(i)

            myWriter.writerow(tmp)

            

from sklearn.neighbors import KNeighborsClassifier  

def knnClassify(trainData,trainLabel,testData): 

    knnClf=KNeighborsClassifier()

    knnClf.fit(trainData,ravel(trainLabel))

    testLabel=knnClf.predict(testData)

    saveResult(testLabel,'sklearn_knn_Result.csv')

    return testLabel

    



from sklearn import svm   

def svcClassify(trainData,trainLabel,testData): 

    svcClf=svm.SVC(C=5.0)   

    svcClf.fit(trainData,ravel(trainLabel))

    testLabel=svcClf.predict(testData)

    saveResult(testLabel,'sklearn_SVC_C=5.0_Result.csv')

    return testLabel

    



from sklearn.naive_bayes import GaussianNB     

def GaussianNBClassify(trainData,trainLabel,testData): 

    nbClf=GaussianNB()          

    nbClf.fit(trainData,ravel(trainLabel))

    testLabel=nbClf.predict(testData)

    saveResult(testLabel,'sklearn_GaussianNB_Result.csv')

    return testLabel

    

from sklearn.naive_bayes import MultinomialNB      

def MultinomialNBClassify(trainData,trainLabel,testData): 

    nbClf=MultinomialNB(alpha=0.1)             

    nbClf.fit(trainData,ravel(trainLabel))

    testLabel=nbClf.predict(testData)

    saveResult(testLabel,'sklearn_MultinomialNB_alpha=0.1_Result.csv')

    return testLabel
def digitRecognition():

    trainData,trainLabel=loadTrainData()

    testData=loadTestData()

    

    result1=knnClassify(trainData,trainLabel,testData)

    result2=svcClassify(trainData,trainLabel,testData)

    result3=GaussianNBClassify(trainData,trainLabel,testData)

    result4=MultinomialNBClassify(trainData,trainLabel,testData)

    

   

    resultGiven=loadTestResult()

    m,n=shape(testData)

    different=0     

    for i in xrange(m):

        if result1[i]!=resultGiven[0,i]:

            different+=1

    print (different)