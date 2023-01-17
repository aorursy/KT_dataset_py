#!wget https://manning-content.s3.amazonaws.com/download/3/29c6e49-7df6-4909-ad1d-18640b3c8aa9/MLiA_SourceCode.zip

#!unzip -o /kaggle/working/MLiA_SourceCode.zip

!ls /kaggle/working/machinelearninginaction
!python -V
# KNN算法

from numpy import *

import operator



def createDataset():

    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])

    labels = ['A', 'A', 'B', 'B']

    return group, labels



def classify0(inX, dataSet, labels, k):

    dataSetSize = dataSet.shape[0]

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat**2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()

    classCount = {}

    for i in range(k):

        voteIlabel = labels[sortedDistIndicies[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]



group, labels = createDataset()

print(classify0([1,1], group, labels, 3))
# KNN算法之应用一：改进约会网站的配对效果

# 1.收集数据，数据已经准备好了，在machinelearninginaction/Ch02/datingTestSet2.txt

!head -3 /kaggle/working/machinelearninginaction/Ch02/datingTestSet2.txt



from numpy import *

import operator

from os import listdir



# 2.准备数据

def file2matrix(filename):

    """

    2.准备数据，从文本中解析数据

    样本数据在machinelearninginaction/Ch02/datingTestSet2.txt中

    包含三种特征：飞行里程数、游戏时间所占百分比、每周消耗冰淇淋的公升数

    """

    fr = open(filename)

    arrayOLines = fr.readlines()

    numberOfLines = len(arrayOLines)

    returnMat = zeros((numberOfLines, 3))

    classLabelVector = []

    index = 0

    for line in arrayOLines:

        line = line.strip()

        listFromLine = line.split('\t')

        returnMat[index, :] = listFromLine[0:3]

        classLabelVector.append(int(listFromLine[-1]))

        index += 1

    return returnMat, classLabelVector



def autoNorm(dataSet):

    """

    2.准备数据，归一化

    """

    minVals = dataSet.min(0)

    maxVals = dataSet.max(0)

    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))

    m = dataSet.shape[0]

    normDataSet = dataSet - tile(minVals, (m, 1))

    normDataSet = normDataSet/tile(ranges, (m, 1))

    return normDataSet, ranges, minVals



# 3. 分析数据

import matplotlib

import matplotlib.pyplot as plt

datingDataMat, classLabels = file2matrix("/kaggle/working/machinelearninginaction/Ch02/datingTestSet2.txt")

normDatingDataMat, ranges, minVals = autoNorm(datingDataMat)

fig = plt.figure()

a1 = fig.add_subplot(131)

a1.scatter(normDatingDataMat[:,0], normDatingDataMat[:, 1], 15.0 * array(classLabels), 15.0 * array(classLabels))

a2 = fig.add_subplot(132)

a2.scatter(normDatingDataMat[:,0], normDatingDataMat[:, 2], 15.0 * array(classLabels), 15.0 * array(classLabels))

a3 = fig.add_subplot(133)

a3.scatter(normDatingDataMat[:,1], normDatingDataMat[:, 2], 15.0 * array(classLabels), 15.0 * array(classLabels))

plt.show()



# 5. 算法测试

def datingClassTest():

    hoRatio = 0.01

    # 训练样本集在machinelearninginaction/Ch02/datingTestSet.txt

    datingDataMat, datingLabels = file2matrix("/kaggle/working/machinelearninginaction/Ch02/datingTestSet2.txt")

    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]

    numTestVecs = int(m * hoRatio)

    errorCount = 0.0

    for i in range(numTestVecs):

        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)

        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))

        if (classifierResult != datingLabels[i]): errorCount += 1.0

    print("the total error rate is : %f" % (errorCount/float(numTestVecs)))



# KNN算法

def classify0(inX, dataSet, labels, k):

    dataSetSize = dataSet.shape[0]

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat**2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()

    classCount = {}

    for i in range(k):

        voteIlabel = labels[sortedDistIndicies[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]



datingClassTest()
# 准备样本集&测试集

!unzip -o /kaggle/working/machinelearninginaction/Ch02/digits.zip -d /kaggle/working/machinelearninginaction/Ch02/



from numpy import *

import operator



# KNN算法

def classify0(inX, dataSet, labels, k):

    dataSetSize = dataSet.shape[0]

    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    sqDiffMat = diffMat**2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()

    classCount = {}

    for i in range(k):

        voteIlabel = labels[sortedDistIndicies[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]



def img2vector(filename):

    """

    将预先处理好的数字文件[32*32]的二维数组转换为[1*1024]的一维数组

    """

    returnVect = zeros((1,1024))

    fr = open(filename)

    for i in range(32):

        lineStr = fr.readline()

        for j in range(32):

            returnVect[0,32*i+j] = int(lineStr[j])

    return returnVect



def handwritingClassTest():

    hwLabels = []

    trainingFileList = listdir('/kaggle/working/machinelearninginaction/Ch02/trainingDigits')           #load the training set

    m = len(trainingFileList)

    trainingMat = zeros((m,1024))

    for i in range(m):

        fileNameStr = trainingFileList[i]

        fileStr = fileNameStr.split('.')[0]     #take off .txt

        classNumStr = int(fileStr.split('_')[0])

        hwLabels.append(classNumStr)

        trainingMat[i,:] = img2vector('/kaggle/working/machinelearninginaction/Ch02/trainingDigits/%s' % fileNameStr)

    testFileList = listdir('/kaggle/working/machinelearninginaction/Ch02/testDigits')        #iterate through the test set

    errorCount = 0.0

    mTest = len(testFileList)

    for i in range(mTest):

        fileNameStr = testFileList[i]

        fileStr = fileNameStr.split('.')[0]     #take off .txt

        classNumStr = int(fileStr.split('_')[0])

        vectorUnderTest = img2vector('/kaggle/working/machinelearninginaction/Ch02/testDigits/%s' % fileNameStr)

        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))

        if (classifierResult != classNumStr): errorCount += 1.0

    print("\nthe total number of errors is: %d" % errorCount)

    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

    

handwritingClassTest()