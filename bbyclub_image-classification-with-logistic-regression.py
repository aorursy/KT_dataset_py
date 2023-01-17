# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from PIL import Image

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

train_cat = "../input/training_set/training_set/cats"

train_dog= "../input/training_set/training_set/dogs"

test_cat= "../input/test_set/test_set/cats"

test_dog= "../input/test_set/test_set/dogs"

image_size = 128
Image.open(train_cat+"/"+"cat.1.jpg")
Image.open("../input/training_set/training_set/dogs/dog.1.jpg")
minh, minv = 100000,100000



for p in range(1,4001):

    pic = Image.open(train_cat+"/"+"cat."+str(p)+".jpg")

    if pic.size[0] < minh:

        minh = pic.size[0]

    if pic.size[1] < minv:

        minv = pic.size[1]

for u in range(1,4001):

    pic = Image.open(train_dog+"/"+"dog."+str(u)+".jpg")

    if pic.size[0] < minh:

        minh = pic.size[0]

    if pic.size[1] < minv:

        minv = pic.size[1]

print(minh)

print(minv)
train_cat_list = []

for p in range(1,4001):

    image = Image.open(train_cat+"/"+"cat."+str(p)+".jpg")

    image = image.resize((minh, minv))

    image = image.convert(mode="L")

    train_cat_list.append(image)

train_dog_list = []

for u in range(1,4001):

    image = Image.open(train_dog+"/"+"dog."+str(u)+".jpg")

    image = image.resize((minh, minv))

    image = image.convert(mode="L")

    train_dog_list.append(image)
x = np.empty((4001+4001, minh * minv))

index = 0

for pl in train_cat_list:

    x[index] = np.array(pl).reshape(minh * minv)

    index += 1

for ul in train_dog_list:

    x[index] = np.array(ul).reshape(minh * minv)

    index += 1    

p = np.ones(4001)

u = np.zeros(4001)

y = np.concatenate((p,u),axis = 0).reshape(x.shape[0],1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T
x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))

x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w, b



def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head



def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients



def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    

    for i in range(number_of_iterarion):

        

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 250 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list



def predict(w,b,x_test):

    

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))



    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction



def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):



    dimension =  x_train.shape[0]

    w,b = initialize_weights_and_bias(dimension)



    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    

    print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100,2)))

    print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100,2)))
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.002, num_iterations = 5001)