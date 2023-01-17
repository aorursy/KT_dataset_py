# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

#from sklearn.model_selection import train_test_split

#from sklearn.preprocessing import StandardScaler

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.metrics import confusion_matrix

#from sklearn.metrics import f1_score

#from sklearn.metrics import accuracy_score

#print(len(Iris))

#print(Iris.head())

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Iris = pd.read_csv("../input/iris/Iris.csv")     #loding dataset

print(Iris)
IrisX=pd.DataFrame(Iris, columns= ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

print(IrisX)
IrisY=pd.DataFrame(Iris, columns= ["Species"])

print(IrisY)
Class={}

for i in range(0,len(Iris["Species"].unique())):

    Class[Iris["Species"].unique()[i]]=i

print(Class)
X = np.array(IrisX, dtype=float).T

Iris['Species']= LabelEncoder().fit_transform(Iris['Species'])

IrisY=pd.DataFrame(Iris, columns= ["Species"])

Y = np.array(IrisY).T

p=np.random.permutation(X.shape[1])

X=X[:,p]

Yorig=Y[:,p]

C=i+1
def oneHotEncoder(array,depth):

    for i in range(0,array.shape[1]):

        a=np.zeros((depth,1))

        a[array[0][i]][0]=1

        new=a if i==0 else np.append(new,a,axis=1)

    return new
Y=oneHotEncoder(Yorig,C)

X_train=X[:,0:140]

Y_train=Y[:,0:140]

YorigTrain=Yorig[:,0:140]

X_test=X[:,140:]

Y_test=Y[:,140:]

YorigTest=Yorig[:,140:]

print(np.shape(X),np.shape(Y),np.shape(X_train),np.shape(Y_train),np.shape(X_test),np.shape(Y_test),np.shape(YorigTrain),np.shape(YorigTest))
def InitializeParameters(Layers):

    parameters={}

    for i in range (len(Layers)-1):

        parameters['W'+str(i+1)]=np.random.randn(Layers[i+1],Layers[i])*np.sqrt(2/(Layers[i]+Layers[i+1]))

        parameters['b'+str(i+1)]=np.zeros((Layers[i+1],1))

    L=int(len(parameters)/2)

    return L,parameters
def RELU(Z):

    return np.maximum(0,Z)
def sigmoid(Z):

    #Z=-np.ones(np.shape(Z))*Z

    return 1/(1+np.exp(-Z))
def ForwardProp(X,parameters,L):

    cache={}

    A=X

    cache['A'+str(0)]=A

    for i in range (L-1):

        Z=np.dot(parameters['W'+str(i+1)],A)+parameters['b'+str(i+1)]

        A=RELU(Z)

        cache['Z'+str(i+1)]=Z

        cache['A'+str(i+1)]=A

        cache['W'+str(i+1)]=parameters['W'+str(i+1)]

        cache['b'+str(i+1)]=parameters['b'+str(i+1)]

    Z=np.dot(parameters['W'+str(L)],A)+parameters['b'+str(L)]

    A=sigmoid(Z)

    cache['Z'+str(L)]=Z

    cache['A'+str(L)]=A

    cache['W'+str(L)]=parameters['W'+str(L)]

    cache['b'+str(L)]=parameters['b'+str(L)]

    return cache,A
def costFunction(arrayLabel,arrayActivated):

    loss=-(1/m)*np.sum(arrayLabel*np.log(arrayActivated)+(1-arrayLabel)*np.log(1-arrayActivated))

    return loss
def sigmoidGrad(array):

    return sigmoid(array)*(1-sigmoid(array))
def RELUGrad(dA,Z):

    dZ=np.array(dA, copy=True)

    dZ[Z<=0]=0

    return dZ
def backProp(X,Y,cache,A):

    L=int(len(cache)/4)

    bCache={}

    dA=-np.divide(Y,A)+np.divide(1-Y,1-A)

    dZ=dA*sigmoidGrad(cache['Z'+str(L)])

    bCache["dW"+str(L)]=(1/m)*np.dot(dZ,cache['A'+str(L-1)].T)

    bCache["db"+str(L)]=(1/m)*np.sum(dZ,axis=1,keepdims=True)

    dA=np.dot(cache['W'+str(L)].T,dZ)

    for i in reversed(range(1,L)):

        dZ=RELUGrad(dA,cache['Z'+str(i)])

        bCache["dW"+str(i)]=(1/m)*np.dot(dZ,cache['A'+str(i-1)].T)

        bCache["db"+str(i)]=(1/m)*np.sum(dZ,axis=1,keepdims=True)

        dA=np.dot(cache['W'+str(i)].T,dZ)

    return bCache
bCache=backProp(X_test,Y_test,cache,A)
def updateParameters(learningRate,Parameters,Gradients):

    for i in range(1,L+1):

        Parameters['W'+str(i)]=Parameters['W'+str(i)]-learningRate*Gradients["dW"+str(i)]

        Parameters['b'+str(i)]=Parameters['b'+str(i)]-learningRate*Gradients["db"+str(i)]

    return Parameters
def deepNeuralNetwork(X,Y,Layers,learningRate):

    L,parameters=InitializeParameters(Layers)

    costs=[]

    for i in range(10000):

        cache,A=ForwardProp(X,parameters,L)

        cost=costFunction(Y,A)

        costs.append(cost)

        Gradients=backProp(X,Y,cache,A)

        parameters=updateParameters(learningRate,parameters,Gradients)

    print ("Cost after iteration %i: %f" %(i, cost))

    plt.plot(costs)

    plt.ylabel('cost')

    plt.xlabel('iterations')

    plt.title("Learning rate =" + str(learningRate))

    plt.show()

    return L,parameters
n,m=X_train.shape

C=3

Layers=[n,5,6,4,2,C]

L,parameters=deepNeuralNetwork(X_train,Y_train,Layers,0.0075)
def predict(X,Y,parameters,L):

    _,A=ForwardProp(X,parameters,L)

    f=np.argmax(A[:,:],axis=0)

    a=np.max(A[0,0])

    s=0

    for i in range(len(f)):

        if Y[0,i]==f[i]:

            s+=1

    return (s/m)*100
prediction=predict(X_train,YorigTrain,parameters,L)

print("Prediction percentage on training set is "+str(prediction)+" %.")
prediction=predict(X_test,YorigTest,parameters,L)

print("Prediction percentage on test set is "+str(prediction)+" %.")