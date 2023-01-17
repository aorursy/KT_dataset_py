import numpy as np
import math
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from numpy import inf
from keras import preprocessing
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
cwd = os.getcwd()
os.chdir(cwd)
print(os.listdir("../input"))

path_cats = []
train_path_cats = "../input/cat-and-dog/training_set/training_set/cats/"
for path in os.listdir(train_path_cats):
    if '.jpg' in path:
        path_cats.append(os.path.join(train_path_cats, path))
path_dogs = []
train_path_dogs = "../input/cat-and-dog/training_set/training_set/dogs/"
for path in os.listdir(train_path_dogs):
    if '.jpg' in path:
        path_dogs.append(os.path.join(train_path_dogs, path))
len(path_dogs), len(path_cats)
# load training set
training_set_orig = np.zeros((6000, 32, 32, 3), dtype='float32')
for i in range(6000):
    if i < 3000:
        path = path_dogs[i]
        img = preprocessing.image.load_img(path, target_size=(32, 32))
        training_set_orig[i] = preprocessing.image.img_to_array(img)
    else:
        path = path_cats[i - 3000]
        img = preprocessing.image.load_img(path, target_size=(32, 32))
        training_set_orig[i] = preprocessing.image.img_to_array(img)
# load test set
test_set_orig = np.zeros((2000, 32, 32, 3), dtype='float32')
for i in range(2000):
    if i < 1000:
        path = path_dogs[i + 3000]
        img = preprocessing.image.load_img(path, target_size=(32, 32))
        test_set_orig[i] = preprocessing.image.img_to_array(img)
    else:
        path = path_cats[i + 2000]
        img = preprocessing.image.load_img(path, target_size=(32, 32))
        test_set_orig[i] = preprocessing.image.img_to_array(img)
training_set_orig.shape
x_train_ = training_set_orig.reshape(6000,-1).T
x_train_.shape
x_test_ = test_set_orig.reshape(2000,-1).T
x_test_.shape
# make target tensor
y_train_ = np.zeros((3000,)) # First 3000 was dog picture so our label is 0
y_train_ = np.concatenate((y_train_, np.ones((3000,)))) # Second 3000 was cat picture so our label is 1
y_test_ = np.zeros((1000,))
y_test_ = np.concatenate((y_test_, np.ones((1000,))))
print("Training set labels" +str(y_train_.shape)+ "  Test set labels" + str(y_test_.shape))
y_train_ = y_train_.reshape(1,-1)
y_test_ = y_test_.reshape(1,-1)
print("Training set labels" +str(y_train_.shape)+ "  Test set labels" + str(y_test_.shape))
np.arange(x_train_.shape[1])

def shuffle_xy(x,y,axis):
    """  Shuffle a two dimensional two array's contents simultaneously in accordance with axis parameter. """   

    
    if (axis == 1 or axis == "columns"):
        c = np.arange(x.shape[1])
        np.random.shuffle(c)
        shuf_x = x[:,c]
        shuf_y = y[:,c]
        shuffled = {"shuffled_x" : shuf_x,
                    "shuffled_y" : shuf_y}
        return shuffled

    
    if (axis == 0 or axis == "rows"):
        r = np.arange(x.shape[0])
        np.random.shuffle(r)
        shuf_x = x[r,:]
        shuffled = {"shuffled_x" : shuf_x}
        return shuffled
        
    else:
        print("Please write an axis argument properly.")    
        
    shuffled = {"shuffled_x" : shuf_x,
                "shuffled_y" : shuf_y}
    return shuffled
    
        
shuffled_dic = shuffle_xy(x_train_,y_train_,1)
x_train_= shuffled_dic["shuffled_x"]
y_train_ = shuffled_dic["shuffled_y"]
x_test = x_test_.copy()
y_test = y_test_.copy()
x_train = x_train_/255
x_test = x_test_/255
index = 2000
plt.imshow(np.uint8(training_set_orig[index]))

N = x_train.shape[0]
#W = np.full((1, x_train.shape[0]),0)
W = np.random.rand(1, x_train.shape[0])/100
grad_past = np.full((1, x_train.shape[0]),0)

b = 0
show = 1 # Python will print loss value every "show" epoch.
e = 0.000001
epoch = 400

W.shape
x_train.shape
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
def propagate(w_p, x_p, b_p, Y_p, alpha, grad_list, momentum, G):
    z = np.dot(w_p, x_p) + b_p

    a = sigmoid(z)
    
    loss = -(1 / N) * np.sum(Y_p * np.log(a) + (1 - Y_p) * (np.log(1 - a))) #Binary cross entropy
    
    dw = (1 / N) * np.dot((a - Y_p), x_p.T)
    #print("dw: ",dw.shape)
    db = (1 / N) * np.sum(a - Y_p)
    learning_rate, G = adagrad2(dw, G)
    W = w_p - (np.array(dw) @ learning_rate)
    #W = w_p - ((momentum * np.array(grad_past)) + (alpha * np.array(dw)))
    #print("W: ", W.shape)    
   # W = w_p - (alpha * np.array(dw))
    b = b_p - alpha*db

    new_values = {"w" : W,
             "b" : b,
             "dw" : dw,
             "cost" : loss}
    return new_values, G
    
    

def training (w_p,X_p, b_p, y_p,alpha,epoch, grad_past_p,momentum_p,show_n_epoch=1):   
    G = 0
    w = w_p[:]
    b = b_p
    x = X_p[:]
    y = y_p [:]
    cost = []
    grad_list = [[grad_past_p]]
    for i in range(epoch):
        shuffled = shuffle_xy(x, y, 1)
        x = shuffled["shuffled_x"]
        y = shuffled["shuffled_y"]
        new_values, G = propagate(w, x , b, y,alpha,grad_list, momentum_p, G)
        grad_list.append(new_values["dw"])
        w = new_values["w"]
        b = new_values["b"]        
        c= new_values["cost"]
        cost.append(c) 
        if i == 0 or i % show_n_epoch == 0:
            print("Epoch :{}  Loss: {}".format(i,c))
    grad = {"W" : w,
            "b" : b,}
            
    
    return grad,cost

def adagrad(dw,G, learning_rate = 0.005):

    dw = np.array((dw))
    dw = dw.reshape(1,-1)
    g = dw.T@dw
    zeros = np.zeros(g.shape, float)
    np.fill_diagonal(zeros, 1)
    g = g * zeros
    G = G + g
    epsilon = np.zeros(G.shape, float)
    np.fill_diagonal(epsilon, 0.000001)
    total = epsilon + G
    total_new = 1/np.sqrt(total)
    total_new[total_new == inf] = 0
    total_new = learning_rate * total_new
    return total_new, G
epsilon = np.zeros((4,4), float)
np.fill_diagonal(epsilon, 0.001)
epsilon
a = 0
epsilon + a
def prediction(w_p, x_p, b_p):
    a = sigmoid(np.dot(w_p, x_p) + b_p)
    Y = ((a >= 0.5).astype(int))
    return Y,a
def accuracy_calculator(W,X,b,y,set_name):
    y_pred = prediction(W,X,b)  
    m = y.shape[1]
    same = (y_pred == y).astype(int)
    acc = 100*(np.sum(same)/m)
    print (" From Scratch Model {} accuracy %{}".format(set_name,acc))
    return acc
def model(w_p,x_p, b_p, Y_p,alpha,epoch,x_test, y_test, grad_past, momentum, show_n_epoch):
    
    grad, cost = training(w_p,x_p, b_p, Y_p, alpha, epoch, grad_past, momentum,show_n_epoch)
    W = grad["W"]
    b = grad["b"]
    
    acc = accuracy_calculator(W,x_test,b_p,y_test,"Test") # To print test accuracy
    accuracy_calculator(W,x_p,b_p,Y_p,"Training") # To print training accuracy

    output = {"acc" : acc,
              "W"    : W,
              "b"    : b, 
              "cost" : cost}
    
    return output
output = model(W, x_train, b, y_train_, alpha, epoch,x_test, y_test,grad_past,momentum,show_n_epoch=10)

cost = output["cost"]
plt.plot(cost)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Loss Curve")
plt.show()
y_train_lib = y_train_.reshape(-1)
y_test_lib = y_test_.reshape(-1)
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 10)
logreg.fit(x_train.T, y_train_lib.T)
print("test accuracy: {} ".format(logreg.score(x_test.T, y_test_lib.T)))
print("train accuracy: {} ".format(logreg.score(x_train.T, y_train_lib.T)))