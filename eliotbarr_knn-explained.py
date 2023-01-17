import pandas as pd

import numpy as np

from scipy.linalg import eigh

import math

import operator

%matplotlib inline



train = pd.read_csv("../input/train.csv")

y_train = train["label"].values

X_train = train.drop('label', axis=1)
def euclideanDistance(x1, x2, length):

    distance = 0

    for x in range(length):

        distance += pow((x1[x] - x2[x]), 2)

    return math.sqrt(distance)
def getNeighbors(trainingSet, testInstance, k):

    distances = []

    length = len(testInstance)-1

    for x in range(len(trainingSet)):

        dist = euclideanDistance(testInstance, trainingSet[x],length)

        distances.append((trainingSet[x], dist))

    distances.sort(key=operator.itemgetter(1))

    neighbors = []

    for x in range(k):

        neighbors.append(distances[x][0])

    return neighbors
def getY(neighbors):

    classVotes={}

    for i in range(len(neighbors)):

        response = neighbors[i][-1]

        if response in classVotes:

            classVotes[response]+=1

        else:

            classVotes[response]=1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    return sortedVotes[0][0]    
class Kernel(object):

	def linear(self):

		def f(x,y):

			return np.dot(x,y)

		return f



	def poly(self,deg0term,degree,gamma):

		def f(x,y):

			return (gamma*np.dot(x,y)+deg0term) ** degree

		return f



	def gaussian(self,gamma):

		def f(x,y):

			return np.exp(-gamma*(np.linalg.norm(x-y)**2))

		return f
def compute_gram(x,kernel_func):

    n_samples,n_features=x.shape

    K=np.zeros([n_samples,n_samples])

    for i,x_i in enumerate(x):

        for j in range(i+1):

            K[i,j]=kernel_func(x_i,x[j,:])

            K[j,i]=kernel_func(x_i,x[j,:])

    return K
def kernel_pca(K,n_components):

    # Performs kernel principal components analysis (kernel PCA)

    #we center the gram matrix

    n = K.shape[0]

    U = np.ones([n,n]) / n

    K_centered= K - U.dot(K) - K.dot(U) + U.dot(K).dot(U)

    # compute eigenvalues and eigenvectors of covariance matrix

    eigvals, eigvecs = eigh(K_centered)

    # Project the data to the new space (k-D) and measure how much variance we kept

    X_new = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))

    return X_new
kernelset = []

value = []



for i in range(1000):

    value = X_train.iloc[i].tolist()

    kernelset.append(value)
valuetest = []



for i in range(1000,1500):

    valuetest = X_train.iloc[i].tolist()

    kernelset.append(valuetest)

    

kernel = Kernel()

func = kernel.poly(deg0term=1,degree= 2,gamma=1/30)

x= np.array(kernelset)
K = compute_gram(x,func)
X_new = kernel_pca(K,n_components = 30)
trainsetkernel = []

valuekernel = []

Ypredict=y_train[:1000]



for i in range(1000):

    valuekernel = X_new[i,:].tolist()

    valuekernel.append(Ypredict[i])

    trainsetkernel.append(valuekernel)
testsetkernel = []

valuekerneltest = []



for i in range(1000,1500):

    valuekerneltest = X_new[i,:].tolist()

    testsetkernel.append(valuekerneltest)
# generate predictions

predictions=[]

k = 5

for x in range(len(testsetkernel)):

    neighbors = getNeighbors(trainsetkernel, testsetkernel[x], k)

    result = getY(neighbors)

    predictions.append(result)
# check error

from sklearn.metrics import accuracy_score

print(accuracy_score(y_train[1000:1500],np.array(predictions)))