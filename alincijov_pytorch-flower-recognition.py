import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import cv2

import os
import torch.nn as nn

import torch.nn.functional as F

import torch
DATADIR = '../input/flowers-recognition/flowers'

CATEGORIES = ["daisy","dandelion","rose","sunflower","tulip"]

IMG_SIZE = 28
def create_training_data():

    training_data = []



    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)

        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):

            try:

                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE)) 

                training_data.append([new_array,class_num])

            except Exception as e:

                pass



    return np.asarray(training_data)
training_data = create_training_data()
features = np.concatenate(training_data[:,0]).reshape(4323, 28, 28).astype(np.float32)

labels = training_data[:,1]
# normalize

features /= 255.



# one hot encoder

labels = np.eye(len(CATEGORIES))[list(labels)]
cuda = torch.device('cuda')
features = torch.from_numpy(features.astype(np.float32)).to(cuda)

labels = torch.from_numpy(labels.astype(np.float32)).to(cuda)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42, shuffle=True)
np.random.seed(34243242)

f = torch.tensor(np.random.randn(3, 3).astype(np.float32)).to(cuda) / 9

w1 = torch.tensor(np.random.randn(169, 64).astype(np.float32)).to(cuda) / 9

w2 = torch.tensor(np.random.randn(64, 32).astype(np.float32)).to(cuda) / 9

w3 = torch.tensor(np.random.randn(32, 5).astype(np.float32)).to(cuda) / 9



theta = f, w1, w2, w3
# Convolution and derivative functions
def conv3x3(x, f):

    conv = torch.tensor(np.zeros((26,26)).astype(np.float32)).to(cuda)

    for i in range(x.shape[0] - 2):

        for j in range(x.shape[1] - 2):

            conv[i,j] = torch.sum(torch.mul(x[i:i+3,j:j+3], f))

    return conv
def dconv(x, f):

    filter = torch.tensor(np.zeros((3,3)).astype(np.float32)).to(cuda)

    for i in range(3):

        for j in range(3):

            filter[i,j] = torch.sum(torch.mul(x[i:i+26,j:j+26], f))

    return filter
def maxpool2x2(x):

    maxpool = torch.tensor(np.zeros((13,13)).astype(np.float32)).to(cuda)

    all_indices = []

    for i in range(0, x.shape[0], 2):

        for j in range(0, x.shape[1], 2):

            values, indices = torch.max(x[i:i+2,j:j+2], 0)

            maxpool[i // 2,j // 2] = values[0]

            indices[0], indices[1] = indices[0] + i, indices[1] + j

            all_indices.append(indices)

    return maxpool, all_indices
# Activation functions
def softmax(x):

    return torch.exp(x)/torch.sum(torch.exp(x))
def dtanh(x):

    return 1.0 - torch.tanh(x)**2
def forward(x, theta):

    f, w1, w2, w3 = theta



    conv = conv3x3(x, f)

    maxpool, maxpool_indices = maxpool2x2(conv)



    flat = maxpool.flatten()

    flat = flat.reshape(1, flat.shape[0])



    p = flat @ w1

    q = torch.tanh(p)

    r = q @ w2

    s = torch.tanh(r)

    t = s @ w3

    u = softmax(t)



    return conv, maxpool, maxpool_indices, flat, p, q, r, s, t, u
def backward(x, y, theta):

    conv, maxpool, maxpool_indices, flat, p, q, r, s, t, u = forward(x, theta)



    e = u - y

    # cross entropy

    # -log(argmax(y))

    error = 1 - ( -torch.log(u[0][torch.argmax(y)]))



    dt = e @ w3.T

    ds = dtanh(r) * dt

    dr = ds @ w2.T

    dq = dtanh(p) * dr

    dp = dq @ w1.T



    dw3 = s.T @ e

    dw2 = q.T @ ds

    dw1 = flat.T @ dq



    dmaxpool = torch.tensor(np.zeros((26,26)).astype(np.float32)).to(cuda)



    for i, p in enumerate(dp.flatten()):

        m, n = maxpool_indices[i]

        dmaxpool[m, n] = p



    # rotate 180

    rot180 = dmaxpool.flip(1)



    df = dconv(x, rot180)



    grads = df, dw1, dw2, dw3



    return grads, error
def update(grads, theta, batch_size, lr=0.05):

    df, dw1, dw2, dw3 = grads

    f, w1, w2, w3 = theta



    f -= (df * lr) / batch_size

    w1 -= (dw1 * lr) / batch_size

    w2 -= (dw2 * lr) / batch_size

    w3 -= (dw3 * lr) / batch_size



    return f, w1, w2, w3
batch_size = 128

torch.backends.cudnn.benchmark = True
for epoch in range(10):

    for idx in np.array_split(np.arange(len(X_train)), len(X_train)/batch_size):

        for i in idx:

            grads, accuracy = backward(X_train[i], y_train[i], theta)

            theta = update(grads, theta, batch_size, lr=0.1)

    print("Epoch:{0:2d}, Accuracy:{1:1.3f}".format(epoch, accuracy))