import pandas as pd
import numpy as np
data = pd.read_csv('/kaggle/input/ex1data2.txt', sep = ',', header = None)

data.info()
data.describe()
data.head()
data = data.rename(columns={0 : 'size', 1 : 'bedrooms', 2 : 'price' })
data.head()
# Mean normalization

dataDescribe = data.describe()
meanValues = dataDescribe.loc['mean', :]
stdValues = dataDescribe.loc['std', :]
minValues = dataDescribe.loc['min', :]
maxValues = dataDescribe.loc['max', :]

meanValues = np.array(meanValues).reshape((3,1))
stdValues = np.array(stdValues).reshape((3,1))

minValues = np.array(minValues).reshape((3,1))
maxValues = np.array(maxValues).reshape((3,1))


diffValues = maxValues - minValues

dataArray = np.array(data)
newData = (dataArray - meanValues.T) / stdValues.T
X = newData[:,[0,1]]
y = newData[:, 2].reshape((X.shape[0],1))
theta0 = np.ones((X.shape[0], 1))
X = np.concatenate((theta0, X), axis = 1)
theta = np.zeros((1, X.shape[1]))
iterations = 1500
alpha = 0.001
m = X.shape[0]
# np.sum(np.multiply(np.dot(X,theta.T) - y, X), axis = 0)
# np.sum(np.multiply(np.dot(X,theta.T) - y, X), axis = 0
# np.multiply(np.dot(X,theta.T) - y, X)
# np.sum(np.multiply(np.dot(X,theta.T) - y, X), axis = 0)
def calculateDelta(X, y):
    return 1/ m * np.sum(np.multiply(np.dot(X,theta.T) - y, X), axis = 0)

for _ in range(iterations):
    theta = theta - alpha * calculateDelta(X,y)
    print (theta)

testSet = np.array([1560, 3])
meanTestSet = meanValues[0:2,:].T
stdTestSet = stdValues[0:2,:].T

# (testSet - meanTestSet)
testSet = (testSet - meanTestSet) / stdTestSet
firstCol = np.array([[1]])
testSet = np.concatenate((firstCol, testSet), axis = 1)
finalVal = np.dot(testSet, theta.T)
meanTestSet = meanValues[2,:]
stdTestSet = stdValues[2,:]

(finalVal * stdTestSet) + meanTestSet
X = dataArray[:,0:2]
onesColumn = np.ones((X.shape[0],1))
X = np.concatenate((onesColumn, X), axis = 1)
y = dataArray[:, 2].reshape((X.shape[0], 1))
pinvMAt = np.linalg.pinv(np.dot(X,X.T))
finalTheta = np.dot(np.dot(pinvMAt,X).T, y)
normalEquTestSet = np.array([[1, 1650, 3]])
np.dot(normalEquTestSet, finalTheta)