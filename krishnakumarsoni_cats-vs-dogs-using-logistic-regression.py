# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
test_dir="/kaggle/input/dogs-cats-images/dataset/test_set"
train_dir = "/kaggle/input/dogs-cats-images/dataset/training_set"

catImages_train = os.listdir('/kaggle/input/dogs-cats-images/dataset/training_set/cats')
print ("Number of Cat images - ",str(len(catImages_train)))

dogImages_train = os.listdir('/kaggle/input/dogs-cats-images/dataset/training_set/dogs')
print ("Number of Dog images - ",str(len(dogImages_train)))

dogImages_test = os.listdir('/kaggle/input/dogs-cats-images/dataset/test_set/dogs')
catImages_test = os.listdir('/kaggle/input/dogs-cats-images/dataset/test_set/cats')
numberOfTestImages = len(dogImages_test) + len(catImages_test)
numberOfTrainingImages = len(catImages_train) + len(dogImages_train)

print("Number of Training Images - " + str(numberOfTrainingImages))
print("Number of Test Images - " + str(numberOfTestImages))
#resizing images for consistency in training and testing
img = cv2.imread(train_dir + '/dogs/dog.1010.jpg')
image = cv2.resize(img, (128, 128))
f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(img)
f.add_subplot(1,2, 2)
plt.imshow(image)
plt.show(block=True)

#train_1x = np.zeros((128*128*3, 1))
train_1y = np.zeros((1, 1))

img1 = cv2.imread(train_dir + '/dogs/dog.101.jpg')
image1 = cv2.resize(img1, (128, 128))
image1 = image1.reshape((128*128*3, 1))
train_1x = image.reshape((128*128*3, 1))
train_1x = np.append(train_1x, image1, axis = 1)

a = train_1x[:,1].reshape((128, 128, 3))
plt.imshow(a)
plt.show()
number = 100
print(image[number][number])
print(a[number][number])
image.shape
#fetching training and test data

WIDTH = 128
HEIGHT = 128

#factor to select lesser number of images
numFac = 0.0625

#train data
train_x = np.zeros((128*128*3, 1))
train_y = np.zeros((1, 1)) 

for i in range(1, 1+int(numFac*(len(catImages_train)))):
    imgCat = cv2.imread(train_dir + '/cats/cat.' + str(i) + '.jpg')
    imgCat = cv2.resize(img, (WIDTH, HEIGHT))
    imgDog = cv2.imread(train_dir + '/dogs/dog.' + str(i) + '.jpg')
    imgDog = cv2.resize(img, (WIDTH, HEIGHT))
    train_x = np.append(train_x, imgCat.reshape(128*128*3, 1), axis = 1)
    train_y = np.append(train_y, np.zeros((1, 1)), axis = 1)
    train_x = np.append(train_x, imgDog.reshape(128*128*3, 1), axis = 1)
    train_y = np.append(train_y, np.ones((1, 1)), axis = 1)
    print(i)
train_x = np.delete(train_x, 0, 1)
train_y = np.delete(train_y, 0, 1)

numberOfTrainingImages = len(train_x[0,:])

#test data
test_x = np.zeros((128*128*3, 1))
test_y = np.zeros((1, 1))  

for i in range(1, 50):
    imgCat = cv2.imread(test_dir + '/cats/cat.' + str(i) + '.jpg')
    imgCat = cv2.resize(img, (WIDTH, HEIGHT))
    imgDog = cv2.imread(test_dir + '/dogs/dog.' + str(i) + '.jpg')
    imgDog = cv2.resize(img, (WIDTH, HEIGHT))
    test_x = np.append(test_x, imgCat.reshape(128*128*3, 1), axis = 1)
    test_y = np.append(test_y, np.zeros((1, 1)), axis = 1)
    test_x = np.append(test_x, imgDog.reshape(128*128*3, 1), axis = 1)
    test_y = np.append(test_y, np.ones((1, 1)), axis = 1)
test_x = np.delete(test_x, 0, 1)
test_y = np.delete(test_y, 0, 1)

numberOfTrainingImages = len(test_x[0,:])

print(len(train_x[:,0]))
print(len(train_y[0,:]))
#Normalize the features
train_x = train_x/225
test_x = test_x/225

def sigmoid(z):
    return 1/(1+np.exp(-z))
#x.shape = (n, m)
#y.shape = (1, m)
#w.shape = (n ,1)
#b.shape = (1, 1)
#z.shape = (1, m)
#L.shape = (1, m)
#dw.shape = (n, 1)
#db.shape = (1, m)
#cost = 

#forward Propagation
#(calculating 'z' and then activation function 'a')
#z = w*x + b
#z.shape = (1, m)
def forwardProp(x, w, b):
    z = np.dot(w.T, x) + b
    return sigmoid(z)

#loss function (L)
#L = -(y*log(y_pred) + (1-y)*log(1-y_pred))
#L.shape = (1, 500)
def loss(x, y, w, b):
    y_pred = forwardProp(x, w, b)
    L =  (-1)*(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)) 
    return L

#cost function (J)
def cost(x, y, w, b):
    return  (1/numberOfTrainingImages)*np.sum(loss(x, y, w, b))

#backProp
#(calculating the gradient for gradient descent algorithm)
#
def backProp(x, y, w, b):
    dw = (1/numberOfTrainingImages)*np.dot(x,((forwardProp(x, w, b)-y).T))
    db = (1/numberOfTrainingImages)*np.sum(forwardProp(x, w, b)-y, axis=1)
    print("-----")
    print(forwardProp(x, w, b)[0, 1:5])
    return dw, db

#gradient descent
"""
Repeat{
w := w + learningRate * dw [where, dw = dJ(w,b)/dw; J = Loss Function] 
b := b + learningRate * db [where, db = dJ(w,b)/db; J = Loss Function]
}
"""

def optimize_GradientDescent(x, y, w, b, numIterations, learningRate):
    
    for i in range(numIterations):        
        dw, db = backProp(x, y, w, b)
        w = w - learningRate * dw
        b = b - learningRate * db
        #print("Cost after iteration" + str(i) + ": " + str(cost(x, y, w, b)))
    return w, b

def predict(x, w, b):
    y_pred = forwardProp(x, w, b)
    for i in range(len(y_pred[:, 0])):
        if y_pred[i, 0] >= 0.5:
            y_pred[i, 0] = 1
        else:
            y_pred[i, 0] = 0
    #assert(y_pred.shape == (1, 500))
    return y_pred

def logisticRegression(train_x, test_x, train_y, test_y, numIterations, learningRate):
    
    #INITIALIZE WEIGHTS AND BIAS
    features = len(train_x[:,0])

    w = np.zeros((features, 1))
    b = np.zeros((1, 1))

    #TRAINING --> get the weights and bias
    w, b = optimize_GradientDescent(train_x, train_y, w, b, numIterations, learningRate)
    
    #PREDICTION 
    y_pred_train = predict(train_x, w, b)
    y_pred_test = predict(test_x, w, b)
    
    #ACCURACY and other metrics
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - Y_test)) * 100))                                       
        
    d = {"Y_pred_test": y_pred_test, 
     "Y_pred_train" : y_pred_train, 
     "w" : w, 
     "b" : b,
     "learningRate" : learningRate,
     "numIterations": numIterations}
    
    return d
#call the model 
model = logisticRegression(train_x, test_x, train_y, test_y, numIterations = 1000, learningRate = 0.005)
a = np.array([[1],
              [3],
              [5]])
b = np.array([[1, 2],
              [3, 4]])
print(np.sum(train_y, axis = 1))
a.shape
train_y.shape
