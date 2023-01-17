# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../working"))

# Any results you write to the current directory are saved as output.
# with open('result.csv') as file:
#     lines = file.readlines()
#     print(lines)
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import time
# load data to matrix
def loadData():
    with open('../input/train.csv') as file:
        lines = file.readlines()[1:]
        trainMat = np.array([list(line.strip().split(',')) for line in lines],dtype=np.int32)
        print(trainMat.shape)
    with open('../input/test.csv') as file:
        lines = file.readlines()[1:]
        testMat = np.array([list(line.strip().split(',')) for line in lines],dtype=np.int32)
        print(testMat.shape)     
    return trainMat[:,1:], trainMat[:,0], testMat

trainData,trainLabels,testData = loadData()
def classify(inputVect,trainImgMat,trainLabels,k):
    distances = np.sqrt(np.sum((inputVect-trainImgMat)**2,1)) # 这里使用了广播
    sortedIndicies = distances.argsort() # 增序
    classCount = {}
    for i in range(k):
        voteLabel = trainLabels[sortedIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0.0) + 1.0
    sortedCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True) # 这里一定要使用.items() 转化成元组列表
    return sortedCount[0][0]
start = time.clock()
resultList = b''
for i in range(testData.shape[0]):
    classifiesResult = classify(testData[i],trainData,trainLabels,3)
    resultList += (b"%d,%d\n"%(i+1,classifiesResult)) # the index of result file starts with 1  
with open('./result.csv','wb+') as wf:
    wf.write(b'ImageId,Label\n')
    wf.write(resultList)
print ("循环用时:", time.clock() - start)