import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2 

import os 

from random import shuffle 

from tqdm import tqdm 

from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import time

import os

from PIL import Image
import os, os.path

train_categories = []

train_samples = []

for i in os.listdir("../input/fruits-360_dataset/fruits-360/Training"):

    train_categories.append(i)

    train_samples.append(len(os.listdir("../input/fruits-360_dataset/fruits-360/Training/"+ i)))



test_categories = []

test_samples = []

for i in os.listdir("../input/fruits-360_dataset/fruits-360/Test"):

    test_categories.append(i)

    test_samples.append(len(os.listdir("../input/fruits-360_dataset/fruits-360/Test/"+ i)))



    

print("No. of Training Samples:", sum(train_samples))

print("No. of Test Samples:", sum(test_samples))





fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 30

fig_size[1] = 5

plt.rcParams["figure.figsize"] = fig_size



index = np.arange(len(train_categories))

plt.bar(index, train_samples)

plt.xlabel('Categories', fontsize=25)

plt.ylabel('No. of samples', fontsize=25)

plt.xticks(index, train_categories, fontsize=15, rotation=90)

plt.title('Category wise training sample distribution', fontsize=35)

plt.show()





index2 = np.arange(len(test_categories))

plt.bar(index2, test_samples)

plt.xlabel('Categories', fontsize=25)

plt.ylabel('No. of samples', fontsize=25)

plt.xticks(index2, test_categories, fontsize=15, rotation=90)

plt.title('Category wise test sample distribution', fontsize=35)

plt.show()
train_b = "../input/fruits-360_dataset/fruits-360/Training/Banana"

train_m= "../input/fruits-360_dataset/fruits-360/Training/Mango"

test_b= "../input/fruits-360_dataset/fruits-360/Test/Banana"

test_m= "../input/fruits-360_dataset/fruits-360/Test/Mango"

image_size = 128
Image.open("../input/fruits-360_dataset/fruits-360/Training/Banana/200_100.jpg")
Image.open("../input/fruits-360_dataset/fruits-360/Training/Mango/200_100.jpg")
for image in tqdm(os.listdir(train_b)): 

    path = os.path.join(train_b, image)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

    img = cv2.resize(img, (image_size, image_size)).flatten()   

    np_img=np.asarray(img)

    

for image2 in tqdm(os.listdir(train_m)): 

    path = os.path.join(train_m, image2)

    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

    img2 = cv2.resize(img2, (image_size, image_size)).flatten() 

    np_img2=np.asarray(img2)



plt.figure(figsize=(10,10))

plt.subplot(1, 2, 1)

plt.imshow(np_img.reshape(image_size, image_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(np_img2.reshape(image_size, image_size))

plt.axis('off')

plt.title("Banana and Mango in GrayScale")
def train_data():

    train_data_b = [] 

    train_data_m=[]

    for image1 in tqdm(os.listdir(train_b)): 

        path = os.path.join(train_b, image)

        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

        img1 = cv2.resize(img1, (image_size, image_size))

        train_data_b.append(img1) 

    for image2 in tqdm(os.listdir(train_m)): 

        path = os.path.join(train_m, image)

        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

        img2 = cv2.resize(img2, (image_size, image_size))

        train_data_m.append(img2) 

    

    train_data= np.concatenate((np.asarray(train_data_b),np.asarray(train_data_m)),axis=0)

    return train_data 

def test_data():

    test_data_b = [] 

    test_data_m=[]

    for image1 in tqdm(os.listdir(test_b)): 

        path = os.path.join(test_b, image1)

        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

        img1 = cv2.resize(img1, (image_size, image_size))

        test_data_b.append(img1) 

    for image2 in tqdm(os.listdir(test_m)): 

        path = os.path.join(test_m, image2)

        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

        img2 = cv2.resize(img2, (image_size, image_size))

        test_data_m.append(img2) 

    

    test_data= np.concatenate((np.asarray(test_data_b),np.asarray(test_data_m)),axis=0) 

    return test_data
train_data = train_data() 

test_data = test_data()
x_data=np.concatenate((train_data,test_data),axis=0)

x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
z1 = np.zeros(490)

o1 = np.ones(490)

Y_train = np.concatenate((o1, z1), axis=0)

z = np.zeros(166)

o = np.ones(166)

Y_test = np.concatenate((o, z), axis=0)
y_data=np.concatenate((Y_train,Y_test),axis=0).reshape(x_data.shape[0],1)
print("X shape: " , x_data.shape)

print("Y shape: " , y_data.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)

number_of_train = x_train.shape[0]

number_of_test = x_test.shape[0]
x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])

x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])

print("X train flatten",x_train_flatten.shape)

print("X test flatten",x_test_flatten.shape)
x_train = x_train_flatten.T

x_test = x_test_flatten.T

y_test = y_test.T

y_train = y_train.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
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

        if i % 100 == 0:

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
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 1500)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

grid={"C":np.logspace(-3,3,7),"penalty":["l1","l2"]},

logistic_regression=LogisticRegression(random_state=42)

log_reg_cv=GridSearchCV(logistic_regression,grid,cv=10)

log_reg_cv.fit(x_train.T,y_train.T)
print("best hyperparameters: ", log_reg_cv.best_params_)

print("accuracy: ", log_reg_cv.best_score_)
log_reg= LogisticRegression(C=1,penalty="l1")

log_reg.fit(x_train.T,y_train.T)

print("test accuracy: {} ".format(log_reg.fit(x_test.T, y_test.T).score(x_test.T, y_test.T)))

print("train accuracy: {} ".format(log_reg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))