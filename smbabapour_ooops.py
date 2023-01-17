

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

trainFile = pd.read_csv('../input/train.csv');

trainFile.head(3)
testFile = pd.read_csv('../input/test.csv');

testFile.head(3)
trainY = trainFile['y']

all = [trainFile, testFile]

allData = pd.concat(all)

allData.head(5)
print('X0: ', pd.Series.unique(allData['X0']))

print('X1: ', pd.Series.unique(allData['X1']))

print('X2: ', pd.Series.unique(allData['X2']))

print('X3: ', pd.Series.unique(allData['X3']))

print('X4: ', pd.Series.unique(allData['X4']))

print('X5: ', pd.Series.unique(allData['X5']))

print('X6: ', pd.Series.unique(allData['X6']))

print('X8: ', pd.Series.unique(allData['X8']))
transformCols = [ 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

allDummies = pd.get_dummies(allData, columns = transformCols)

allDummies.head(3)
allColumns = list(set(allDummies.columns))



for column in allColumns:

    nElements = len(np.unique(allDummies[column]))

    if nElements == 1:

        allDummies.drop(column, axis=1)



allDummies.head(5)
allrIdy = allDummies.drop(['ID', 'y'], axis=1)

allrIdy
from sklearn.decomposition import TruncatedSVD

np.random.seed(5)

svd = TruncatedSVD(n_components=100)

svd.fit(allrIdy)

allSVD = svd.transform(allrIdy)
trainData = allSVD[:len(trainFile)]

trainData
testData = allSVD[len(testFile):]

testData
trainX = trainData

testX = testData
from xgboost.sklearn import XGBRegressor



xreg = XGBRegressor()

xreg.fit(trainX, trainY)  

pred = xreg.predict(testX)
predList = pred.tolist()

predList
testId = testFile['ID'].values

testY = predList

predFile = pd.DataFrame()

predFile['ID'] = testId

predFile['y'] = testY

predFile.to_csv('XGBRegressor.csv', index=False)

predFile