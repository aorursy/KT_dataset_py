%matplotlib inline
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import sklearn.neural_network as nn

plt.ion()   # interactive mode
import pickle
print(os.listdir("../input"))
train_imgs = pickle.load(open("../input/CPSC340FinalPart2/train_images_512.pk",'rb'), encoding='bytes')
train_imgs = pickle.load(open("../input/CPSC340FinalPart2/train_images_512.pk",'rb'), encoding='bytes')
train_labels = pickle.load(open("../input/CPSC340FinalPart2/train_labels_512.pk",'rb'), encoding='bytes')
test_imgs = pickle.load(open("../input/CPSC340FinalPart2/test_images_512.pk",'rb'), encoding='bytes')

class CovidDatasetTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

class CovidDatasetTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx]  
class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        N, D = self.X.shape
        y_pred = np.zeros(Xtest.shape[0])
        
        #X_dist = euclidean_dist_squared(Xtest, self.X)
        distArray = cosine_vectorized(Xtest, self.X)
        X_dist = np.ones(distArray.shape) - distArray

        for i in range(X_dist.shape[0]):
            x_row = np.argsort(X_dist[i])
            x_k = x_row[:self.k]
            #print("HIII",self.y[np.array(x_k).item()])
            count = np.bincount(self.y[x_k])
            y_pred[i] = np.argmax(count)
        return y_pred
def cosine_vectorized(array1, array2):
    sumyy = (array2**2).sum(1)
    sumxx = (array1**2).sum(1, keepdims=1)
    sumxy = array1.dot(array2.T)
    return (sumxy/np.sqrt(sumxx))/np.sqrt(sumyy)
def make_data_loaders():
    train_dataset = CovidDatasetTrain(train_imgs, train_labels)
    test_dataset = CovidDatasetTest(test_imgs)

    return {
        "train": DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1),
        "test": DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1),
    }


"""
Saves predictions as a csv file in proper format to be submited to Kaggel
    input: y_pred - results of predictions, file_name - name of csv file to be saved
"""
def save_results(y_pred, file_name):
    n = y_pred.shape[0]
    y_i = range(1,n+1) 
    y_out = np.append(np.transpose([y_i]),np.transpose([y_pred]),axis=1)

    np.savetxt(file_name, y_out, delimiter=',',fmt=['%d', '%d'], header='Id,Predicted', comments='')

data_loaders = make_data_loaders()
dataset_sizes = {'train': len(data_loaders['train'].dataset), 
                 'test':len(data_loaders['test'].dataset)}

class_names = ['covid', 'background']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Training loop starter
num_epochs = 1 # Set this yourself

for epoch in range(num_epochs):
    for sample in data_loaders["train"]:
        pass
    # Image shape
    # Batch size x Channels x Width x Height
    print(sample[0].shape)
    #print(sample.shape)
    # Labels shape
    # Batch size
    print(sample[1].shape)

print("hi", train_imgs.shape)
X = []
Xtest = []
for sample in train_imgs:
    #print(sample.shape)
    X.append(np.array(sample).reshape([1*3*512*512]))
    
for sample in test_imgs:
    #print(sample.shape)
    Xtest.append(np.array(sample).reshape([1*3*512*512]))
X = np.array(X)
Xtest = np.array(Xtest)
y = np.array(train_labels)
#ytest = test_labels
#print(X)
model = KNN(k=5)
model.fit(X, y)
#print(X)
y_pred = model.predict(Xtest) #change this
v_error = np.mean(y_pred != y) #change this
print("KNN (ours) training error: %.3f" % v_error)
save_results(y_pred,'CPSC340_Q2_SUBMISSION_KNN.csv') # save y_pred as csv file
model = nn.MLPClassifier(hidden_layer_sizes=(100,50,))

model.fit(X, y)
y_pred = model.predict(Xtest)
save_results(y_pred,'CPSC340_Q2_SUBMISSION_MLP.csv') # save y_pred as csv file