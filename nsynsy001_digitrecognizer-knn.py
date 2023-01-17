#coding:utf-8
import numpy as np
import pandas as pd
import operator

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

#获取训练数据
def getDigitData():
	mTrain = train.shape[0]
	#训练集train
	trainer = train[0 : int(mTrain * 0.9)]
	#测试集test
	tester = train[int(mTrain * 0.9) : mTrain]
	trainData = trainer.values;
	X = trainData[:, 1:trainData.shape[1]]
	y = trainData[:, 0]
	return X, y, tester

#kNN算法
def classify(inX, dataSet, labels, k):
	m = dataSet.shape[0]
	diffMat = np.tile(inX, (m, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistance = sqDiffMat.sum(axis = 1)
	distance = sqDistance ** 0.5
	sortedDistIndicies = distance.argsort()
	classCount = {}
	
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def pridictDigit():
	testData = test.values
	X, y, t = getDigitData()
	for i in testData:
		predictNum = classify(i, X, y, 3)
		print(predictNum)
		

pridictDigit()
# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs