import os
print(os.listdir("../input"))
# Designed by Junbo Zhao
# 12/23/2013
# Processing the Training and Testing data. Writing them into .txt files.

from numpy import *

#You can use this simple data to Debug and validate this program
def loadSimpleData():
    data = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    label = [1,1,-1,-1,1]
    return data, label


# Generate random data for Debug and validation.
def randomData(fname,sampleNum,option):
    import random
    if not isinstance(fname,basestring):
        print ('File cannot be built!')
        return
    filename = fname + '.txt'
    fid = open(filename,'w')
    if option == 'train':
        for i in range(sampleNum):
            fid.write('+1:%.4f,%.4f,%.4f,%.4f,%.4f\n' %(random.random(),random.random(),random.random(),random.random(),random.random()))
            fid.write('-1:%.4f,%.4f,%.4f,%.4f,%.4f\n' %(random.random(),random.random(),random.random(),random.random(),random.random()))
        fid.close()
        return
    elif option == 'test':
        for i in range(sampleNum):
            fid.write('%.4f,%.4f,%.4f,%.4f,%.4f\n' %(random.random(),random.random(),random.random(),random.random(),random.random()))
            fid.write('%.4f,%.4f,%.4f,%.4f,%.4f\n' %(random.random(),random.random(),random.random(),random.random(),random.random()))
        fid.close()
        return
    else:
        print ('Wrong input parameter!')
        return

    
# Read data from Training or Testing file
def readData(filename,option):
    if option == 'train':
        fid = open(filename,'r')
        print ('Preparing training data')
        label = []
        data = None
        while True:
            fline = fid.readline()
            if len(fline) == 0:     #EOF
                break
            label.append(int(fline[0:fline.find(':')]))
            
            dataNew = []
            i = fline.find(':') + 1
            dataNew = [float(fline[i:fline.find(',',i,-1)])]
            while True:
                i = fline.find(',',i,-1) + 1
                if not i:
                    break
                dataNew.append(float(fline[i:fline.find(',',i,-1)])) # Excellent design of python!!! No problem of this!
            if data is None:
                data = mat(dataNew)
            else:
                data = vstack([data,mat(dataNew)])
        fid.close()
        return data,label
    elif option == 'test':
        fid = open(filename,'r')
        print ('Preparing training data')
        data = None
        while True:
            fline = fid.readline()
            if len(fline) == 0:     #EOF
                break
            
            dataNew = []
            i=0
            while True:
                dataNew.append(float(fline[i:fline.find(',',i,-1)])) # Excellent design of python!!! No problem of this!
                i = fline.find(',',i,-1) + 1
                if not i:
                    break
            if data is None:
                data = mat(dataNew)
            else:
                data = vstack([data,mat(dataNew)])
        fid.close()
        return data
    else:
        print ('Wrong input parameter!')
# Designed by Junbo Zhao
# 12/23/2013
# This .py file includes the adaboost core codes.


# Building weak stump function
def buildWeakStump(d,l,D):
    dataMatrix = mat(d)
    labelmatrix = mat(l).T
    m,n = shape(dataMatrix)
    numstep = 10.0
    bestStump = {}
    bestClass = mat(zeros((5,1)))
    minErr = inf
    for i in range(n):
        datamin = dataMatrix[:,i].min()
        datamax = dataMatrix[:,i].max()
        stepSize = (datamax - datamin) / numstep
        for j in range(-1,int(numstep)+1):
            for inequal in ['lt','gt']:
                threshold = datamin + float(j) * stepSize
                predict = stumpClassify(dataMatrix,i,threshold,inequal)
                err = mat(ones((m,1)))
                err[predict == labelmatrix] = 0
                weighted_err = D.T * err
                if weighted_err < minErr:
                    minErr = weighted_err
                    bestClass = predict.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshold
                    bestStump['ineq'] = inequal
    return bestStump, minErr, bestClass

# Use the weak stump to classify training data
def stumpClassify(datamat,dim,threshold,inequal):
    res = ones((shape(datamat)[0],1))
    if inequal == 'lt':
        res[datamat[:,dim] <= threshold] = -1.0
    else:
        res[datamat[:,dim] > threshold] = -1.0
    return res

# Training
def train(data,label,numIt = 1000):
    weakClassifiers = []
    m = shape(data)[0]
    D = mat(ones((m,1))/m)
    EnsembleClassEstimate = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEstimate = buildWeakStump(data,label,D)
        alpha = float(0.5*log((1.0-error) / (error+1e-15)))
        bestStump['alpha'] = alpha
        weakClassifiers.append(bestStump)
        weightD = multiply((-1*alpha*mat(label)).T,classEstimate)
        D = multiply(D,exp(weightD))
        D = D/D.sum()
        EnsembleClassEstimate += classEstimate*alpha
        EnsembleErrors = multiply(sign(EnsembleClassEstimate)!=mat(label).T,\
                                  ones((m,1)))  #Converte to float
        errorRate = EnsembleErrors.sum()/m
        print ("Error Rate:",errorRate)
        if errorRate == 0.0:
            break
    return weakClassifiers

# Applying adaboost classifier for a single data sample
def adaboostClassify(dataTest,classifier):
    dataMatrix = mat(dataTest)
    m = shape(dataMatrix)[0]
    EnsembleClassEstimate = mat(zeros((m,1)))
    for i in range(len(classifier)):
        classEstimate = stumpClassify(dataMatrix,classifier[i]['dim'],classifier[i]['threshold'],classifier[i]['ineq'])
        EnsembleClassEstimate += classifier[i]['alpha']*classEstimate
        #print EnsembleClassEstimate
    return sign(EnsembleClassEstimate)

# Testing
def test1(dataSet,classifier):
    label = []
    print ('\n\n\nResults: ')
    for i in range(shape(dataSet)[0]):
        label.append(adaboostClassify(dataSet[i,:],classifier))
        print('%s' %(label[0]))
    return label
# Designed by Junbo Zhao
# 12/23/2013
# This is a simple demo, training and testing the adaboost classifier

from numpy import *


# The following data files are only used to show you how this program works.
# The two files are generated by function randomData()

# It is better to use your own data to see the power of adaboost!
# Face recognition problems are good for using adaboost.

trainData,label = readData('../input/train.txt','train')
#trainData,label = data.loadSimpleData()
testData = readData('../input/test.txt','test')
classifier = train(trainData,label,150)
test1(testData,classifier)



