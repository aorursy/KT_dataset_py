import numpy as np

import scipy

import sklearn

import os

import urllib3

import csv



import math
X=[]

print(os.listdir("../input"))



c=1

with open("../input/mnist-original/mnist_train.csv", 'r') as csvf:

    csvr=csv.reader(csvf)

    for row in csvr:

        X.append(row)

Y=[]

for i in range(60000):

    Y.append(X[i][0])

    del(X[i][0])

X=np.array(X,dtype="float")

X[0]
X=X/255.0

X=X.T

Y=np.array(Y,dtype="int")

Y=Y.T

Y

X.shape
Y1=np.zeros((10,60000),dtype=np.int8)

for i in range(60000):

    Y1[int(Y[i])][i]=1

Y=Y1

Y.shape

h=2 #number of hidden lyers

nodes=[X.shape[0],64,32,Y.shape[0]] #number of nodes in these hidden layers
W=[]

B=[]

for i in range(h+1):

    W.append(np.random.rand(nodes[i+1],nodes[i])*0.01)

    B.append(np.random.rand(nodes[i+1],1)*0.01)

#W.append(np.random.rand(Y.shape[0],nodes[h-1])*0.001)

print(W[0].shape,B[0].shape)

print(W[1].shape,B[1].shape)

print(W[2].shape,B[2].shape)

b=[]

def tanh(x):

    return((math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)))

def dtanh(x):

    tanh_vec=np.vectorize(tanh)

    return 1-tanh_vec(x)**2
def fp(X,Y):

    Z=[]

    A=[]

    A.append(X)

    for i in range(h+1):

        Z.append(np.dot(W[i],A[i]))

        Z[i]=Z[i]+B[i]

        #tanh_vec=np.vectorize(tanh)

        A.append(np.tanh(Z[i]))

    return Z,A
Z,A=fp(X,Y)

for i in range(3):

    print(Z[i].shape,A[i].shape)

print(A[3].shape)
def bp(X,Y):

    m=X.shape[1]

    dz=[None]*(h+1)

    dw=[None]*(h+1)

    db=[None]*(h+1)

    #dtanh_vec=np.vectorize(dtanh)

    Z,A=fp(X,Y)

    dz[h]=((1/m)*(A[h+1]-Y)*(1-np.tanh(Z[h])*np.tanh(Z[h])))

    #print("her1")

    dw[h]=np.dot(dz[h],A[h].T)

    db[h]=np.sum(dz[h],axis=1,keepdims=True)

    i=h-1

    #print("here")

    while i>=0:

        #print("heer")

        dz[i]=np.dot(W[i+1].T,dz[i+1])*(1-np.tanh(Z[i])*np.tanh(Z[i]))

        dw[i]=np.dot(dz[i],A[i].T)

        db[i]=np.sum(dz[i],axis=1,keepdims=True)

        i=i-1

    return dw,db
import random

batch_size=1000

for i in range(2000):

    for j in range(X.shape[1]//batch_size):

        indices=random.sample([i for i in range(X.shape[1])],batch_size)

        X_sample,Y_sample=X.T[indices].T,Y.T[indices].T

        dw,db=bp(X_sample,Y_sample)

        for k in range(h+1):

            #print("weight update",i,j)

            W[k]=W[k]-0.01*dw[k]

            B[k]=B[k]-0.01*db[k]

    if(i%10==0):

        Z,A=fp(X,Y)

        temp=A[h+1].T

        count=0

        for cc in range(X.shape[1]):

            max1=0

            for cc1 in range(Y.shape[0]):

                if(temp[cc][cc1]>temp[cc][max1]):

                    max1=cc1

            if(Y.T[cc][max1]==1):

                count=count+1

        print("count=",count)

    print("epoch=",i)
Z,A=fp(X,Y)
temp=A[3].T

count=0

for i in range(60000):

    max1=0

    for j in range(10):

        if(temp[i][j]>temp[i][max1]):

            max1=j

    if(Y.T[i][max1]==1):

        print(max1)

        count=count+1

count
A[3].T