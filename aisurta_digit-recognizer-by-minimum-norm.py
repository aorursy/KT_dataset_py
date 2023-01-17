from numpy import savetxt, loadtxt

from sklearn.externals import joblib

dataset = loadtxt(open('train.csv', 'r'), dtype='f8', delimiter=',', skiprows=1)
testset = loadtxt(open('test.csv', 'r'), dtype='f8', delimiter=',', skiprows=1)

joblib.dump(dataset, 'training_set.pkl')
joblib.dump(testset, 'test_set.pkl')


dataset = joblib.load('training_set.pkl')
testset = joblib.load('test_set.pkl')

target = [x[0] for x in dataset[:42000]]

train = [x[1:] for x in dataset[:42000]]

test = [x[0:] for x in testset[:28000]]

targetarr = [[0 for i in range(10)] for j in range(42000)]

for i in range(0, 42000):
	targetarr[i][int(target[i])] = 1;
	
import numpy as np

trainT = list(map(list, zip(*train)))

trainNum = np.array([train])
trainTNum = np.array([trainT])
targetarrNum = np.array([targetarr])
testNum  =np.array([test])

S = trainTNum[0][:][:].dot(trainNum[:][:][0])

A = np.linalg.lstsq(trainNum[0][:][:],targetarrNum[0][:][:])[0]

out =testNum.dot(A)

(out.T).argmax(0)

np.savetxt('Submit_MMNRM.csv', np.c_[range(1, len(test) + 1), (out.T).argmax(0)], delimiter=',', comments = '', header = 'ImageId,Label', fmt='%d')