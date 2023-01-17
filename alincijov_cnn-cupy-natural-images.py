import cupy as cp

import numpy as np
import pandas as pd

from tqdm import tqdm

import os

import cv2

import math

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from cupyx.scipy.ndimage.filters import convolve

import skimage.measure
path = '../input/natural-images/natural_images/'
idx_elements = { k:v for (k,v) in enumerate(list(os.walk(path))[0][1])}

elements_idx = { v:k for (k,v) in enumerate(list(os.walk(path))[0][1])}
features = []

labels = []



for folder in list(os.walk(path))[1:]:

    feature = folder[0].split('/')[-1]

    for img_path in folder[2]:

        features.append(cv2.resize(cv2.imread(path + feature + "/" + img_path), (28, 28)))

        one_hot = cp.zeros(len(idx_elements))

        one_hot[elements_idx[feature]] = 1



        labels.append(one_hot)



features = cp.array(features)

labels = cp.array(labels)
print(features.shape)
# normalize

features = features / 255.0
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
def relu(x):

    mask = (x>0) * 1.0 

    return x * mask



def drelu(x):

    mask = (x>0) * 1.0 

    return  mask
def sigmoid(z):

  return 1.0 / (1 + cp.exp(-z))



def dsigmoid(z):

  return sigmoid(z) * (1-sigmoid(z))
def softmax(s): 

    exps = cp.exp(s - cp.max(s, axis=1, keepdims=True))

    return exps/cp.sum(exps, axis=1, keepdims=True)



def cross_entropy(pred, real):

    n_samples = real.shape[0]

    res = pred - real

    return res/n_samples
def error(pred, real):

    n_samples = real.shape[0]

    logp = - cp.log(pred[cp.arange(n_samples), real.argmax(axis=1)])

    loss = cp.sum(logp)/n_samples

    return loss
def forward(x, theta):

    k, w1, w2, b1, b2 = theta



    m = cp.asarray([convolve(input=x[i], weights=k) for i in range(len(x))])

    n = relu(m)

    o = cp.asarray([skimage.measure.block_reduce(cp.asnumpy(n[i]), (2,2,1), np.max) for i in range(len(m))])

    f = o.reshape(n.shape[0], 588)

    p = f.dot(w1) + b1

    q = sigmoid(p)

    r = q.dot(w2) + b2

    s = softmax(r)



    return m, n, o, f, p, q, r, s
def backward(x, y, theta):

    k, w1, w2, b1, b2 = theta

    m, n, o, f, p, q, r, s = forward(x, theta)



    ds = cross_entropy(s, y)

    dr = ds.dot(w2.T)

    dq = dr * dsigmoid(p)

    dp = dq.dot(w1.T)



    db2 = cp.mean(ds, axis=0)

    dw2 = ds.T.dot(q).T



    db1 = cp.mean(dq, axis=0)

    dw1 = dq.T.dot(f).T



    masks = cp.asarray([cp.equal(n[i], o[i].repeat(2, axis=0).repeat(2, axis=1)).astype(int) for i in range(len(n))])

    windows = cp.asarray([(masks[i] * dp[i].reshape(14, 14, 3).repeat(2, axis=0).repeat(2, axis=1)) for i in range(len(masks))])



    dk = cp.mean(cp.array([cp.rot90(convolve(x[i], cp.rot90(windows[i] * n[i],2 )),2) for i in range(len(windows))]))



    return dk, dw1, dw2, db1, db2
def optimize(grads, theta, lr=0.1):

    theta = tuple([theta[i] - (grads[i] * lr) for i in range(len(theta))])

    return theta
cp.random.seed(35435345353)



k = cp.random.uniform(size=(28, 28, 3))



w1 = cp.random.uniform(size=(588, 64))

b1 = cp.random.uniform(size=(1, 64))



w2 = cp.random.uniform(size=(64, 8))

b2 = cp.random.uniform(size=(1, 8))



theta = k, w1, w2, b1, b2
# mini batches

x_batches = cp.array_split(X_train, math.ceil(len(X_train) / 500))

y_batches = cp.array_split(y_train, math.ceil(len(y_train) / 500))
errors = []
for epoch in range(301):

    for i,x_batch in enumerate(x_batches):

        grads = backward(x_batch, y_batches[i], theta)

        theta = optimize(grads, theta, 0.005)



    if(epoch % 25 == 0):

        e = error(forward(x_batches[0], theta)[-1], y_batches[0])

        errors.append(e)

        print('Epoch:{0}, Error:{1}'.format(epoch, e))
plt.plot(errors)