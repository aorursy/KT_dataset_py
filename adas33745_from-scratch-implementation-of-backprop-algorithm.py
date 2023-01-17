#!conda env list
#import sys

#!conda install --yes --prefix {sys.prefix} scikit-learn
# PACKAGE

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cbook

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

#import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
# input methods

import os

print(os.listdir("../input"))

training_data = pd.read_csv("../input/train.csv")

training_data = training_data.values
length = training_data.shape[0]

length

training_length = round(0.7*(length))

#dev_length = round(0.8*(length))

x_train = training_data[:training_length,1:].T

y_train = training_data[:training_length,0]



#x_dev = training_data[training_length+1:dev_length,1:].T

#y_dev = training_data[training_length+1:dev_length,0]



x_test = training_data[training_length+1:,1:].T

y_test = training_data[training_length+1:,0]







y_train = y_train.reshape(1,len(y_train))

#y_dev = y_dev.reshape(1,len(y_dev))

y_test = y_test.reshape(1,len(y_test))
scaler = MinMaxScaler(feature_range=(0, 1))

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test) 
#scaler_train = StandardScaler().fit(x_train)

#scaler_test = StandardScaler().fit(x_test)

#x_train = scaler_train.transform(x_train)

#x_test = scaler_test.transform(x_test)


img = x_train[:,3]

img = img.reshape(28,28)

plt.imshow(img,cmap = 'gray')

plt.show()

#y_train[:,3]
count_in_labels = []

for i in range(10):

    idx = y_train == i

    count_in_labels.append(len(y_train[idx]))

    print("Count in label %d is %d"%(i,count_in_labels[i]))
logisticregsr = LogisticRegression(multi_class='multinomial', solver = 'lbfgs')

y_train_l = y_train.ravel()

logisticregsr.fit(x_train.T,y_train_l)
pred = logisticregsr.predict(x_test.T)

y_test_check = y_test.ravel()

y_train_check = y_train.ravel()

#y_test_check.shape

score_train = logisticregsr.score(x_train.T,y_train_check)*100

print("The accuracy on train data is",score_train,"%")

score_test = logisticregsr.score(x_test.T,y_test_check)*100

print("The accuracy on test data is",score_test,"%")
c_matrix = confusion_matrix(y_test_check,pred)
print(c_matrix)
def transform_output(y):

    l = y.shape[1]

    ty = np.zeros((10,l))

    for i in range(l):

        s = y[0,i].astype(int)

        # print(type(s))

        ty[s,i] = 1

    return ty
def function(z):

    return np.maximum(0.01*z,z)
def function_prime(z):

    a = np.greater(z,0.01*z)

    a = a.astype(int)

    a[a==0] = 0.01

    return a
def cost_function(y,a,m):

    b = y-a

    s = (1/(2*m))*(np.dot(b.T,b))

    l = s.shape[1]

    dsum = 0

    for i in range(l):

        dsum = dsum + s[i,i]

    return dsum
def sigmoid(z):

    return 1/(1 + np.exp(-z))
def sigmoid_prime(z):

    return sigmoid(z)*(1-sigmoid(z))
def grad_a_cost_function(y,a,m):

    return -(1/m)*(y-a)

class Network:

    # atype is the activation type; whether it is a RELU or sigmoid

    def __init__(self,sizes,atype):

        self.num_layers = len(sizes)

        self.weights = []

        if atype=="sigmoid":

            self.biases = [np.random.randn(y,1) for y in sizes[1:]]

            #self.biases = [np.zeros((y,1)) for y in sizes[1:]]

            #self.weights = [np.zeros((sizes[1],sizes[0]))]

            #self.weights = [np.zeros((sizes[1],sizes[0]))

            for x in range(1,len(sizes)):

                (self.weights).append(np.random.randn(sizes[x],sizes[x-1]))

        else:

            self.biases = [np.zeros((y,1)) for y in sizes[1:]]

            #self.weights = [np.zeros((sizes[1],sizes[0]))]

            for x in range(1,len(sizes)):

                (self.weights).append(np.random.randn(sizes[x],sizes[x-1])*np.sqrt(2/sizes[x-1]))

                

            

    def feedforward(self, batch_x):

        length = len(self.weights)

        a_cache=[]

        a = batch_x

        a_cache.append(a)

        z_cache=[] 

        for i in range(length-1):

            z = np.dot(self.weights[i],a) + self.biases[i]

            z_cache.append(z)

            a = function(z)

            a_cache.append(a)

        i=i+1

        z = np.dot(self.weights[i],a) + self.biases[i]

        z_cache.append(z)

        a = sigmoid(z)

        a_cache.append(a)

        return a_cache,z_cache

    

    def feedforward_output(self, x_test):

        length = len(self.weights)

        a = x_test

        for i in range(length-1):

            z = np.dot(self.weights[i],a) + self.biases[i]

            a = function(z)

            #print("z[",i,"] = ",z)

        i = i + 1

        z = np.dot(self.weights[i],a) + self.biases[i]

        #print("z[",i,"] = ",z)

        a = sigmoid(z)

        return a

    

    def compute_dZ(self,batch_x,batch_y,l,Z_cache,dZ):

        m = batch_x.shape[1]

        n = batch_x.shape[0]

        #y = batch[n-1,:]

        #y = y.reshape(1,len(y))

        # print(type(y))

        #print(y.shape)

        #y = transform_output(y)

        if l== self.num_layers - 2:

            #a = function(Z_cache[l])

            a = sigmoid(Z_cache[l])

            #return grad_a_cost_function(batch_y,a,m)*function_prime(Z_cache[l]) 

            return grad_a_cost_function(batch_y,a,m)*sigmoid_prime(Z_cache[l])

        else:

            return np.dot(self.weights[l+1].T,dZ)*function_prime(Z_cache[l])

    

    def backpropagation(self,batch_x,batch_y):

        dw = []

        db = []

        m = batch_x.shape[1]

        A_cache,Z_cache = self.feedforward(batch_x)

        L = self.num_layers

        dZ = []

        for l in range(L-2,-1,-1):

            dZ = self.compute_dZ(batch_x,batch_y,l,Z_cache,dZ)

            db.append((1/m)*np.sum(dZ,axis = 1,keepdims = True))

            dw.append((1/m)*np.dot(dZ,A_cache[l].T))

        db.reverse()

        dw.reverse()

        return dw,db 

        

    def update(self,batch_x,batch_y,eta,vdw,vdb):

        beta = 0.9

        dw,db = self.backpropagation(batch_x,batch_y)

        

        vdw = [beta*x +(1-beta)*y for (x,y) in zip(vdw,dw)]

        vdb = [beta*x +(1-beta)*y for (x,y) in zip(vdb,db)]

        for i in range(self.num_layers-1):

            #check(dw[i])

            self.weights[i] = self.weights[i] - eta*vdw[i]

            self.biases[i] = self.biases[i] - eta*vdb[i]

        

        return vdw,vdb



            

    # we assume that in the training data, rows represent the features of the training set and columns denote the 

    # instances of the training set.

    

    def SGD(self,x_train,y_train,batch_size,no_of_epoch,eta,x_test=None,y_test=None):

        training_data = x_train

        print(training_data.shape)

        print(y_train.shape)

        training_data = np.append(training_data,y_train,axis = 0)

        print(training_data.shape)

        # In the previous two lines, we have put x_train and y_train into one matrix: training data;

        #training_data = [[x_train],

        #                 [ytrain]]

        n = training_data.shape[0]

        training_size = training_data.shape[1]

        vdw = [np.zeros(y.shape) for y in self.weights]

        vdb = [np.zeros(y.shape) for y in self.biases]

        for i in range(no_of_epoch):

            np.random.shuffle(training_data.T)

            x_train_shuffled = training_data[:-1,:]

            y_train_shuffled = training_data[n-1,:]

            y_train_shuffled = y_train_shuffled.reshape(1,len(y_train_shuffled ))

            # print(type(y_train_shuffled))

            y_train_shuffled_transformed = transform_output(y_train_shuffled)

            batches_x = [x_train_shuffled[:,k:k + batch_size] for k in range(0,training_size,batch_size)]

            batches_y = [y_train_shuffled_transformed[:,k:k + batch_size] for k in range(0,training_size,batch_size)]

            for (batch_x,batch_y) in zip(batches_x,batches_y):

                vdw,vdb = self.update(batch_x,batch_y,eta,vdw,vdb)

            a = self.feedforward_output(x_train[:,1:10])

            m = a.shape[1]

            y = transform_output(y_train[:,1:10])

            cost = cost_function(y,a,m)

            print("The cost after epoch no.",i," is ",cost)

            #a = np.argmax(a,axis = 0)

            #s = score(y_train,a)

            #print(s)

        

        
n = Network([784,30,30,10],"RELU")

n.SGD(x_train,y_train,10,50,0.5)
img = x_test[:,7]
img = img.reshape(len(img),1)

A_test= n.feedforward_output(img)

A_test = np.argmax(A_test,axis=0)

print(img.shape)

img = img.reshape(28,28)

plt.imshow(img,cmap = 'gray')

plt.show()

print("The predicted output is", A_test)
A_train = n.feedforward_output(x_train)

A_train = np.argmax(A_train,axis = 0)

#A.shape

A_train = A_train.reshape(1,len(A_train))
A_test = n.feedforward_output(x_test)

A_test = np.argmax(A_test,axis = 0)

A_test.shape

A_test = A_test.reshape(1,len(A_test))
def score(y,a):

    l = y.shape[1]

    return (1 - (np.count_nonzero(y-a)/l))*100
score_train = score(y_train,A_train)

score_test = score(y_test,A_test)

print("Accuracy on train data is ",score_train, "% and on test data is",score_test,"%")