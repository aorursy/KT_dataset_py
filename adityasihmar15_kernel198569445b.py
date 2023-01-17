# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:53:14 2020

@author: aditya
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from PIL import Image
from numba import jit, cuda
import tensorflow as tf
from scipy import optimize
import torch.optim as optim
import random
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(gpu)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 4 )
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 4)
        self.max_pl = nn.MaxPool2d(2,2)
        
    def foreward(self, imgs):
        x = torch.tensor(imgs, device=gpu).float()
        x = self.max_pl(F.relu(self.conv1(x)))
        x = self.max_pl(F.relu(self.conv2(x)))
        x = self.max_pl(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], -1)
        x = F.sigmoid(nn.Linear(in_features=x.size()[-1], out_features = 6)(x))
        return x
    
        
net = Network().to(gpu)
optimizer = optim.Adam(net.parameters(), lr = 0.00001)
loss_function = nn.CrossEntropyLoss()
            
def load(path):
  imgs = []
  labels = []
  label = -1
  n = 0
  for sbf in os.listdir(path):
    label = label + 1
    n = 0
    pth = os.path.join(path, sbf)
    for ssbf in tqdm(os.listdir(pth)):
      n = n + 1
      img = cv2.imread(os.path.join(pth, ssbf),0)
      img = cv2.resize(img, img_size)
      imgs.append((img/255).reshape(img_size[0],img_size[1],1))
      labels.append(label)

  tmp = list(zip(imgs, labels))
  random.shuffle(tmp)
  imgs, labels = zip(*tmp)
  return np.array(imgs), np.array(labels)

def clc_X_batch(X, theta, y):
    '''This function is for calculating the approximate value of X-batch for computing the error in convolutional layers'''
    X = y.dot(np.linalg.pinv(theta))
    return torch.tensor(X, device = gpu).float()

def gradient_des_convolution(X, y, theta, lr, lmbda=0.1):
    h = hypothesis(X, theta)
    J = compute_cost(X, y, theta)
    theta -= lr/m * np.transpose(X).dot(h-y) + (lmbda/m)*np.sum(theta[0:-1])
    batch_y_cnv = clc_X_batch(X, theta, y)
    X = torch.tensor(X, device = gpu).float()
    X.requires_grad = True
    net.zero_grad()
    optimizer.zero_grad()
    #outputs = net.forward(X)

    loss = loss_function(X, batch_y_cnv)
    loss.backward()
    optimizer.step()
    return theta, J, loss

def hypothesis(X, theta):
    return sigmoid(X.dot(theta))

def convolution_py(imgs, batch_size):
    x = net.foreward(imgs)
    return x

def convolution_tf(imgs, batch_size):
    input_shape = [batch_size, imgs.shape[0], imgs.shape[1], imgs.shape[2]]
    x = tf.convert_to_tensor(imgs)
    #print(x.shape)
    x = tf.keras.layers.Conv2D(32, 4, input_shape = [batch_size, imgs.shape[0], imgs.shape[1], imgs.shape[2]], activation = 'relu')(x)
    x  = tf.keras.layers.MaxPool2D((2,2))(x)
    #print(x.shape)
    x = tf.keras.layers.Conv2D(64, 4, activation = 'relu')(x)
    x  = tf.keras.layers.MaxPool2D((2,2))(x)
    #print(x.shape)
    #x = tf.keras.layers.Conv2D(128, 4, activation = 'relu')(x)
    #x  = tf.keras.layers.MaxPool2D((2,2))(x)
    #print(x.shape)
    #x = tf.keras.layers.Conv2D(128, 4, activation = 'relu')(x)
    #x  = tf.keras.layers.MaxPool2D((2,2))(x)
    #print(x.shape)
    x = tf.keras.layers.Conv2D(1, 4, activation = 'relu')(x)
    x  = tf.keras.layers.MaxPool2D((2,2))(x)
    #print(x.shape)
    x = tf.keras.layers.Flatten()(x)
    #print(x.shape)
    return  x.numpy()

def batch(btach_size, theta, X, y, alpha, iterations,lmbda):
    J_hst = []
    cnn_loss = []
    for itr in tqdm(range(iterations)):
        #tmp = list(zip(X, y))
        #random.shuffle(tmp)
        #X, y = zip(*tmp)
        #tmp = None
        for i in range(0,len(y),batch_size):
            xb = X[i:i+batch_size, :, :, :]
            #print(xb.shape)
            yb = y[i:i+batch_size]
            xb = net.foreward(xb)
            if i == 0:
                theta = np.zeros([xb.shape[1], 1])
            theta, J_hist, loss = gradient_des_convolution(xb, yb, theta, alpha)
            J_hst.append(J_hist)
            cnn_loss.append(loss)
        
    return theta, J_hst, cnn_loss

def compute_cost(X, y, theta):
    #print('theta shape = ',  theta.shape)
    #print('X shape= ', X.shape)
    h = hypothesis(X, theta)

    J = -1/(m) * np.sum(np.transpose(y).dot(np.log(h)) + np.transpose(1-y).dot(np.log(1-h)))
    return J

def flatten(arr):
    arr = tf.convert_to_tensor(arr)
    arr = tf.keras.layers.Flatten()(arr)
    return arr.numpy()

def gradient_descent(theta, X, y, lr, iteration, lmbda):
    
    h = hypothesis(X, theta)
    theta -= lr/m * np.transpose(X).dot(h-y) + (lmbda/m)*np.sum(theta[0:-1])
    J = compute_cost(X, y, theta)

    return theta, J

def compute_gradient(theta, X, y):
    h = hypothesis(X, theta)
    #print(h.shape)
    #print(np.transpose(X).shape)
    grad = 1/m * np.transpose(X).dot(h-y)
    return grad

def sigmoid(z):
    return 1/(1+np.exp(-z))


def predict(all_theta, X):
  y_ = sigmoid(X.dot(all_theta.T))
  #print(y_.shape)
  return np.argmax(y_, axis = 1), y_


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path_train = '/kaggle/input/intel-image-classification/seg_train/seg_train'
path_test = '/kaggle/input/intel-image-classification/seg_test/seg_test'
path_pred = '/kaggle/input/intel-image-classification/seg_pred/seg_pred'
img_size = (120,120)

train_dirs = os.listdir(path_train)
test_dirs = os.listdir(path_test)

X, y = load(path_train)
m = X.shape[0]
#X = np.concatenate((np.ones([X.shape[0], 1], dtype='float64'), X), axis=1)
print('\n X shape = ', X.shape)
print('\n Y shape = ', y.shape)
#test_data = load(path_test)
theta = np.zeros([img_size[0]*img_size[1], 1])
print('theta shape = ', theta.shape)

alpha = 0.000001
iterations = 20
lmbda = 0
all_theta = []
all_J = []
cnn_loss = []
X = X.reshape(m, 1, img_size[0],img_size[1])
batch_size = 400
for i in range(6):
    theta, J_hist, loss = batch(batch_size, theta, X, (y==i).reshape(m, 1), alpha, iterations,lmbda)
    cnn_loss.append(loss)
    all_theta.append(np.squeeze(theta))
    all_J.append(J_hist)
    theta = np.zeros([img_size[0]*img_size[1], 1])
all_theta = np.array(all_theta)
print('all_theta shape = ', all_theta.shape)
acc_hist = []
for i in range(0, X.shape[0], batch_size):
    x = net.foreward(X[i:i+batch_size,:,:,:])
    y_pred, y_ = predict(all_theta, x)
    acc = np.mean(y[i: i+batch_size] == y_pred) * 100
    acc_hist.append(acc)
    print('accuracy = ', acc) 
    
print('Average Accuracy = ', np.mean(acc_hist))

plt.plot(all_J[0])
plt.plot(cnn_loss[0])
m = nn.Conv2d(3,32,4)
for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)

model_parameters = filter(lambda p: p.requires_grad, m.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
for param in net.parameters():
    #print(type(param.grad))
    param.data = 0 * param.data
def batch(btach_size, theta, X, y, alpha, iterations,lmbda):
    J_hst = []
    cnn_loss = []
    for itr in tqdm(range(iterations)):
        #tmp = list(zip(X, y))
        #random.shuffle(tmp)
        #X, y = zip(*tmp)
        #tmp = None
        for i in range(0,len(y),batch_size):
            xb = X[i:i+batch_size, :, :, :]
            #print(xb.shape)
            yb = y[i:i+batch_size]
            output = net.foreward(xb)
            net.zero_grad()
            optimizer.zero_grad()
            #outputs = net.forward(X)

            loss = loss_function(output, yb)
            loss.backward()
            optimizer.step()
            cnn_loss.append(loss)
        
    return cnn_loss

cnn_loss = batch(batch_size, None, X, y, alpha, iterations, None)

