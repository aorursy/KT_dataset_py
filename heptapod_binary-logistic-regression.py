# Importing the required libraries (plus no heavy use of scikit-learn):



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn.utils import shuffle
# Processing the data:



def get_data():

	titanic = pd.read_csv("../input/train_and_test2.csv")

	data = titanic.as_matrix()

	X = data[:,1:-1]

	Y = data[:,-1]

	X[:,1] = (X[:,1]-X[:,1].mean())/X[:,1].std()

	X[:,0] = (X[:,0]-X[:,0].mean())/X[:,0].std()

	N, D = X.shape

	X2 = np.zeros((N,D))

	X2[:,0:3] = X[:,0:3]

    

	# One-Hot-Encoding:

	for n in range(N):

		t = int(X[n,D-3]) #Embarked

		t2 = int(X[n,D-6]) #pclass

		t3 = int(X[n,D-15]) #parch

		t4 = int(X[n,D-23]) #sibsp

		X2[n,t+D-3] = 1

		X2[n,t2+D-6] = 1

		X2[n,t3+D-15] = 1

		X2[n,t4+D-23] = 1



		return X2, Y



def get_binary_data():

	X, Y = get_data()

	X2 = X[Y <= 1]

	Y2 = Y[Y <= 1]

	return X2, Y2
X, Y = get_binary_data()

X, Y = shuffle(X,Y)



Xtrain = X[0:891,:]

Ytrain = Y[0:891]

Xtest = X[-418:]

Ytest = Y[-418:] # Which I manually put to be zero in every row!



D = X.shape[1]

W = np.random.randn(D)

b = 0
# Making some necessary functions:



def sigmoid(z):

	return 1/(1+np.exp(-z))



def forward(X,W,b):

	return sigmoid(X.dot(W)+b)



def classification_rate(targets,predictions):

	return np.mean(targets == predictions)



def cross_entropy(T,pY):

	return -np.mean(T*np.log(pY) + (1-T)*np.log(1-pY))
# Logistic regression via gradient descent plus L2 regularization:



train_costs = []

test_costs = []

learning_rate = 0.001



for i in range(10000):

	pYtrain = forward(Xtrain,W,b)

	pYtest = forward(Xtest,W,b)



	ctrain = cross_entropy(Ytrain,pYtrain)

	ctest = cross_entropy(Ytest,pYtest)

	train_costs.append(ctrain)

	test_costs.append(ctest)



	W -= learning_rate*(Xtrain.T.dot(pYtrain-Ytrain)-0.1*W)

	b -= learning_rate*(pYtrain-Ytrain).sum()
# Displaying my model:



legend1, = plt.plot(train_costs,label='Train Cost')

legend2, = plt.plot(test_costs,label='Test Cost')

plt.legend([legend1,legend2])

plt.show()



print("Final Train Classification Rate: ",classification_rate(Ytrain,np.round(pYtrain)))

print("Final Test Classification Rate: ",classification_rate(Ytest,np.round(pYtest)))