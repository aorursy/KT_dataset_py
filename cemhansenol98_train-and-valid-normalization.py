import torch

import torch.nn as nn

from PIL import Image

from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

import os

import math

import pandas as pd
PATH = Path('../input/kodluyoruz-mist/data/mnist')
img = np.array(Image.open("../input/kodluyoruz-mist/data/mnist/train/0/img_11161.jpg"))
img.shape
plt.imshow(img, cmap = "gray");
kernel = np.array([-1,1])
out = np.zeros((28,27))
img.shape
def conv(img, kernel):

    

    out = np.zeros(img.shape)

    img = np.pad(img,[(0, 0), (0, 1)],"edge") # This will do the padding for not to reduce size

    

    for i in range(img.shape[0]):

    

        for j in range(img.shape[1]-1):

            out[i][j] = abs((img[i][j:j+2] * kernel).sum())

    

    return out
plt.imshow(img, cmap = "gray");
out = conv(img, kernel)
plt.imshow(out, cmap = "gray");
def _get_files(p, fs, extensions = None):

    p = Path(p) # to support / notation

    res = [p/f for f in fs if not f.startswith(".") 

           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]

    return res
def create_ds_from_file(src):

    imgs, labels = [], []

    

    for label in range(10):

        path = src/str(label)

        print(path)

        t = [o.name for o in os.scandir(path)]

        t = _get_files(path, t, extensions = [".jpg", ".png"])

        for e in t:

            l = [np.array(Image.open(e)).reshape(28*28)]

            imgs += l

        labels += ([label] * len(t))

    return torch.tensor(imgs,  dtype=torch.float32), torch.tensor(labels, dtype=torch.long).view(-1,1)
trn_x, trn_y = create_ds_from_file(PATH/"train")
val_x,val_y = create_ds_from_file(PATH/"validation")
def normalization(array):

    return (array - array.min()) / (array.max() - array.min())
img_norm = normalization(img)
plt.imshow(img_norm, cmap = "gray");
trn_x.shape
def multiple_normalization(train_or_valid_X):

    for i in range(len(train_or_valid_X)):

        train_or_valid_X[i] =  (train_or_valid_X[i] - torch.min(train_or_valid_X[i])) / (torch.max(train_or_valid_X[i]) - torch.min(train_or_valid_X[i]))

    return train_or_valid_X
train_norm = multiple_normalization(trn_x)
train_norm[0].shape
plt.imshow(train_norm[0].reshape(28,28), cmap = "gray");
val_x.shape
valid_norm = multiple_normalization(val_x)
valid_norm[0].shape
valid_norm[0].reshape(28,28)
plt.imshow(valid_norm[0].reshape(28,28), cmap = "gray");
def create_ds_from_file(src):

    imgs, labels = [], []

    

    for label in range(10):

        path = src/str(label)

        print(path)

        t = [o.name for o in os.scandir(path)]

        t = _get_files(path, t, extensions = [".jpg", ".png"])

        for e in t:

            img = np.array(Image.open(e))

            l = [np.concatenate((conv(img, kernel).reshape(-1), img.reshape(-1)))]

            imgs += l

        labels += ([label] * len(t))

    return torch.tensor(imgs,  dtype=torch.float32), torch.tensor(labels, dtype=torch.long).view(-1,1)
trn_x, trn_y = create_ds_from_file(PATH/"train")
val_x,val_y = create_ds_from_file(PATH/"validation")
plt.imshow(trn_x[0].view(56,28), cmap = "gray")
trn_x[0].shape
plt.imshow(trn_x[0].reshape(56,28)[:28,:28], cmap = "gray")
plt.imshow(trn_x[0].reshape(56,28)[28:,:28], cmap = "gray")
trn_x[0].shape[0]
def stack_normalization(array):

    # Filtreli kısım icin

    array_f = array[:784]

    for i in range(784):

        filtre_array = (array_f - array_f.min()) / (array_f.max() - array_f.min())

        

    # Filtresiz kısım icin

    array_fs = array[784:]

    for j in range(784,1568):

        filtresiz_array = (array_fs - array_fs.min()) / (array_fs.max() - array_fs.min())

    

    arr = np.hstack((filtre_array,filtresiz_array))

        

    return np.array(arr)
stack_normalization(trn_x[0])
stack_normalization(trn_x[0]).shape
plt.imshow(trn_x[0].view(56,28), cmap = "gray")
plt.imshow(stack_normalization(trn_x[0]).reshape(56,28), cmap = "gray")