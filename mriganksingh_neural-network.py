# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import h5py

from PIL import Image

from scipy import ndimage

from sklearn import preprocessing

np.random.seed(1)
def load_dataset():

    train_dataset = h5py.File('../input/train_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels



    test_dataset = h5py.File('../input/test_catvnoncat.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels



    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

x_train, y_train, x_test, y_test, classes = load_dataset()

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

x_train=x_train/255

x_test=x_test/255
#not a cat image

plt.imshow(x_train[0])

print("Value of y in this case is : "+str(y_train[0][0]))
#cat image

plt.imshow(x_train[2])

print("Value of y in this case is : "+str(y_train[0][2]))
#reshaping the image

x_train=x_train.reshape((209,64*64*3))

print(x_train.shape)

print(x_train[0,:])
x_test=x_test.reshape((50,64*64*3))

x_test.shape
#Feature_scaling

scaler=preprocessing.StandardScaler()

scaler.fit(x_train)

x_train=scaler.transform(x_train)

x_test=scaler.transform(x_test)
#First applying inbuilt Logistic Regression

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(solver='lbfgs',max_iter=1000)

clf.fit(x_train,y_train.reshape(209))
y_predict=clf.predict(x_test)

print("Test score :  "+str(clf.score(x_test,y_test.reshape(50))))

print("Train score :  "+str(clf.score(x_train,y_train.reshape(209))))
x_train=x_train.reshape((12288,209))

x_test=x_test.reshape((12288,50))
#defining the layers of neural network

def layer_size(x_train,y_train):

    n_x=x_train.shape[0]

    n_h=4

    n_y=y_train.shape[0]

    return n_x,n_h,n_y
#initialising the parameters

def initialize_parameters(n_x,n_h,n_y):

    np.random.seed(2)

    w1=np.random.randn(n_h,n_x)*0.01

    b1=np.zeros((n_h,1))

    w2=np.random.randn(n_y,n_h)*0.01

    b2=np.zeros((n_y,1))

    parameters={"w1":w1,

               "w2":w2,

               "b1":b1,

               "b2":b2}

    return parameters
def sigmoid(z):

    return 1/(1+np.exp(-z))
def forward_propagation(x,parameters):

    w1=parameters["w1"]

    w2=parameters["w2"]

    b1=parameters["b1"]

    b2=parameters["b2"]

    z1=np.dot(w1,x)+b1

    a1=np.tanh(z1)

    z2=np.dot(w2,a1)+b2

    a2=sigmoid(z2)

    cache={"z1":z1,

          "z2":z2,

          "a1":a1,

          "a2":a2}

    return a2,cache
#cost

def compute_cost(a2,y,parameters):

    m=y.shape[1]

    loss=np.multiply(np.log(a2),y)+np.multiply(np.log(1-a2),1-y)

    cost=-(np.sum(loss))/m

    return cost
#backward propagation

def backward_propagation(parameters,cache,x,y):

    w1=parameters["w1"]

    w2=parameters["w2"]

    b1=parameters["b1"]

    b2=parameters["b2"]

    z1=cache["z1"]

    z2=cache["z2"]

    a1=cache["a1"]

    a2=cache["a2"]

    m=x.shape[1]

    dz2=a2-y

    dw2=(np.dot(dz2,a1.T))/m

    db2=(np.sum(dz2,axis=1,keepdims=True))/m

    dz1=(np.multiply(np.dot(w2.T,dz2),1-np.power(a1,2)))

    dw1=np.dot(dz1,x.T)/m

    db1=(np.sum(dz1,axis=1,keepdims=True))/m

    grads={"dw1":dw1,

           "dw2":dw2,

           "db1":db1,

           "db2":db2}

    return grads     
#updating parameters

def update_parameters(parameters,grads,learning_rate=0.01):

    w1=parameters["w1"]

    b1=parameters["b1"]

    w2=parameters["w2"]

    b2=parameters["b2"]

    dw1=grads["dw1"]

    db1=grads["db1"]

    dw2=grads["dw2"]

    db2=grads["db2"]

    w1=w1-(learning_rate*dw1)

    w2=w2-(learning_rate*dw2)

    b1=b1-(learning_rate*db1)

    b2=b2-(learning_rate*db2)

    parameters={"w1": w1,

                "w2": w2,

                "b1": b1,

                "b2": b2}

    return parameters    
def nn_model(x,y,n_h,num_iterations=2000):

    n_x=layer_size(x,y)[0]

    n_y=layer_size(x,y)[2]

    parameters=initialize_parameters(n_x,n_h,n_y)

    w1=parameters["w1"]

    b1=parameters["b1"]

    w2=parameters["w2"]

    b2=parameters["b2"]

    for i in range(0,num_iterations):

        a2,cache=forward_propagation(x,parameters)

        cost=compute_cost(a2,y,parameters)

        grads=backward_propagation(parameters,cache,x,y)

        parameters=update_parameters(parameters,grads)

        #if(i%1000==0):

         #   print("Cost after iteration %i: %f" % (i, cost))

    return parameters    
def predict(parameters,x):

    a2, cache = forward_propagation(x, parameters)

    predictions = np.round(a2)

    return predictions
parameters=nn_model(x_train,y_train,n_h=4)
predictions = predict(parameters, x_test)

predictions1 = predict(parameters, x_train)
print ('Accuracy: %d' % float((np.dot(y_test, predictions.T) + np.dot(1 - y_test, 1 - predictions.T)) / float(y_test.size) * 100) + '%')

print ('Accuracy: %d' % float((np.dot(y_train, predictions1.T) + np.dot(1 - y_train, 1 - predictions1.T)) / float(y_train.size) * 100) + '%')