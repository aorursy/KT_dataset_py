# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.chdir("../input")

# Any results you write to the current directory are saved as output.
#import data
full_data = pd.read_csv("data.csv")
#drop ID
full_data = full_data.drop('id', 1)
#numerize diagnosis - 0 benign 1 malign
full_data['diagnosis'] = full_data['diagnosis'].map({'M':1 , 'B':0}).astype(int)

full_data = full_data.drop('Unnamed: 32', 1)

print(full_data.describe())
from sklearn.model_selection import train_test_split
train, test = train_test_split(full_data, test_size=0.2, shuffle=True)
train, test = np.array(train).astype(np.float64), np.array(test).astype(np.float64)
print(train.shape, test.shape)
#useful values
m_train = train.shape[0]
m_test = test.shape[0]
n = 30

train_X = train[:,1:]
train_y = train[:,0].reshape(m_train, 1)
test_X = test[:,1:]
test_y = test[:,0].reshape(m_test, 1)

for full_data in [train_X, test_X]:
    full_data = (full_data - full_data.mean()) / (full_data.max() - full_data.min())

print(train_X.shape, train_y.shape)
def sigmoid(x):
    # input: 
    # x, np array of shape (n,m)
    # output: s, np array of shape (n,m)
    s = 1/(1+np.exp(-x))
    return s
def loss(a, y, m):
    # input: a, y are np array of shape (1,m)
    # output: l, np array of shape (1,1)
    l = np.sum(np.dot(y, np.log(a+1).T) + np.dot((1-y), np.log(1-a+1).T))
    l *= -1/m
    l += np.log(2)
    return l
#set up the right dimensions
train_X, train_y, test_X, test_y = map(lambda x: x.T, [train_X, train_y, test_X, test_y])
#dimensional check
assert(train_X.shape==(n,m_train))
print(train_X, train_y)
import time as time
#training
"""def train(train_X, train_y, w, b, lr, n_iter, loss_log):
    for iter in range(n_iter):
        z = np.dot(np.transpose(w), train_X) + b #(1,n)*(n,m)=(1,m)
        a = sigmoid(z) #(1,m)
        L = loss(a, train_y, m_train) #(1,1)
        np.append(loss_log, L)
        dz = a-train_y #(1,m)
        dw = np.dot(train_X, np.transpose(dz)) #(n,m)*(m,1)=(n,1)
        w -= lr*dw
        b -= lr

def main(train_iter, test_intv, lr, train_log, val_log):
    iter_count = 0
    #initialize all matrices
    z=np.zeros((1,m))
    
    tic = time.time()
    print("Training started. %i iterations, testing every %i iterations. Learning rate=%i" % (train_iter, test_intv, lr))
    while(iter_count<train_iter):
        train(train_X, train_y, w, b, lr, test_intv, train_log)
        toc = time.time()
        iter_count += test_intv
        print("Iteration %i. Time taken = %5f , loss = %5f" % (iter_count, toc-tic, L))
""" #kill all function style for now


"""
#train loop
for iter in range(n_iter):
    z = np.dot(np.transpose(w), train_X) + b #(1,n)*(n,m)=(1,m)
    a = sigmoid(z) #(1,m)
    L = loss(a, train_y, m_train) #(1,1)
    np.append(loss_log, L)
    dz = a-train_y #(1,m)
    dw = np.dot(train_X, np.transpose(dz)) #(n,m)*(m,1)=(n,1)
    w -= lr*dw
    b -= lr

#test loop
for iter in range(n_iter):
    z = np.dot(np.transpose(w), test_X) + b #(1,n)*(n,m)=(1,m)
    a = sigmoid(z) #(1,m)
    L = loss(a, test_y, m_test) #(1,1)
    np.append(test_log, L)
"""
#hyperparameters
n_iter = 10000
test_intv = 100
summary = 20
lr = 0.75

#initialize
z_train = np.random.rand(1,m_train)
a_train = np.random.rand(1,m_train)
z_test = np.random.rand(1,m_test)
a_test = np.random.rand(1,m_test)
train_log = [[]]
test_log = [[]]
b = np.random.rand(1,1)
w = np.random.rand(n, 1)

#main loop
iter_count = 0
tic = time.time()
print("Training started. %i iterations, testing every %i iterations. Learning rate=%5f" % (n_iter, test_intv, lr))
for i in range(n_iter+1):
    z_train = np.dot(np.transpose(w), train_X) + b #(1,n)*(n,m)=(1,m)
    a_train = sigmoid(z_train) #(1,m)
    L_train = loss(a_train, train_y, m_train) #(1,1)
    np.append(train_log, L_train)
    dz = a_train-train_y #(1,m)
    dw = np.dot(train_X, np.transpose(dz)) #(n,m)*(m,1)=(n,1)
    w -= lr*dw
    b -= lr
    if iter_count%summary==0:
        toc=time.time()
        print("Iteration %i. Time = %5f , training loss = %s" % (iter_count, toc-tic, str(L_train)))
    if iter_count%test_intv==0:#test
        z_test = np.dot(np.transpose(w), test_X) + b #(1,n)*(n,m)=(1,m)
        a_test = sigmoid(z_test) #(1,m)
        L_test = loss(a_test, test_y, m_test) #(1,1)
        np.append(test_log, L_test)
        toc=time.time()
        print("Validation at iteration %i. Time = %5f , validation loss = %s" % (iter_count, toc-tic, str(L_test)))
    iter_count += 1
print(np.append(a_test.T, test_y.T, axis=1))
import matplotlib.pyplot as plt
plt.plot(train_log)
plt.show()
