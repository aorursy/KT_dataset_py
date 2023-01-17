# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib notebook

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from sklearn.decomposition import PCA
from math import *
featurex = np.genfromtxt('../input/PCA.txt', delimiter=',')
featurex.shape
featurex = featurex.T
feature = np.genfromtxt('../input/feature.txt', delimiter=',')
feature1 = feature[0, :]
feature2 = featurex[0, :]
feature3 = featurex[1, :]
feature4 = featurex[2, :]
feature5 = featurex[3, :]
stest = 127038
strain = 400000
train1 = feature1[0:-stest]
train2 = feature2[0:-stest]
train3 = feature3[0:-stest]
train4 = feature4[0:-stest]
train5 = feature5[0:-stest]

teach = feature1[1:-stest+1]

test1 = feature1[strain:]
test2 = feature2[strain:]
test3 = feature3[strain:]
test4 = feature4[strain:]
test5 = feature5[strain:]


train = np.vstack((train1, train2, train3, train4, train5))
train = train.T

test = np.vstack((test1, test2, test3, test4, test5))
test = test.T
print(train.shape)
print(train1.shape, test1.shape)
def crossVaild(Winsf, Wsf, bsf, alpha):
    kfold = 10
    wash = 10
    _input = 5
    _output = 1
    units = 100
    activate = 0
    Win = np.genfromtxt('../input/Winp.txt')
    W = np.genfromtxt('../input/Wp.txt')
    bias = np.genfromtxt('../input/biasp.txt')
    
    Win *= Winsf
    W *= Wsf
    bias *= bsf

    NRMSE_vec = np.zeros(kfold)
    MAPE_vec = np.zeros(kfold)
    Nmax = np.shape(train)[0]
    selection = 0

    for k in range(kfold):
        aval = (k) * int(np.floor(Nmax/kfold))
        bval = (k+1) * int(np.floor(Nmax/kfold))
        print('ROUND: ', k)

        input_test = train[aval:bval, :]
        nteach = np.hstack((teach[0:aval],teach[bval:Nmax]))
        ntrain = np.vstack((train[0:aval, :],train[bval:Nmax, :]))

        trainlen = np.shape(ntrain)[0]
        testlen = np.shape(input_test)[0]


        # data collection matrix
        SCM = np.zeros((_input+units,trainlen), dtype=np.float32)
        #p = np.zeros((1+units+5), dtype=np.float32)
        x = np.zeros((units,), dtype=np.float32)
        u = np.zeros((_input,), dtype=np.float32)
        print(u.shape)
        # training phase
        for j in range(trainlen):
            u = ntrain[j, :]
            #p[0:5] = u
            #print(Win.shape, u.T.shape, W.shape, x.shape, bias.shape)
            x = np.tanh(np.dot(Win,u.T) + np.dot(W,x) + bias)
            #p[5:] = x
            #SCM[:, j] = p
            x = np.expand_dims(x, axis=1)
            u = np.expand_dims(u, axis=1)
            #print(u.shape, x.shape)
            SCM[:, j] =  np.vstack((u, x))[:,0]
            x = np.squeeze(x, axis=1)
            u = np.squeeze(u, axis=1)
            #break


        nteach = nteach[wash:]
        SCM = SCM[:, wash:]

        Wout = np.dot(np.dot(nteach.T, SCM.T), np.linalg.inv(np.dot(SCM,SCM.T) + alpha*np.eye(units+_input)))

        # Prediction phase
        Y = np.zeros(testlen)
        u = input_test[0, :]

        for t in range(testlen-1):
            x = np.tanh(np.dot(Win,u.T) + np.dot(W,x) + bias)
            #p[0] = u
            #p[1:] = x
            x = np.expand_dims(x, axis=1)
            u = np.expand_dims(u, axis=1)
            y = np.dot(Wout, np.vstack((u, x)))
            #y = np.dot(Wout, p)
            x = np.squeeze(x, axis=1)
            u = np.squeeze(u, axis=1)
            Y[t] = y
            u = input_test[t+1, :]

        # Error calculation
        in_test = input_test[0:, selection]
        Y = Y[0:testlen]
        Error = in_test - Y
        MSE = np.mean(np.power(Error, 2))
        varD = np.mean(np.power((in_test - np.mean(in_test)), 2))
        NRMSE = sqrt(MSE/varD)
        NRMSE_vec[k] = NRMSE
    meanNRMSE = np.mean(NRMSE_vec)
    print('mean: ', meanNRMSE, '\nNrmse: ', NRMSE_vec)
    return meanNRMSE

def run(Winsf, Wsf, bsf, alpha):
    error = 100
    newerror = crossVaild(Winsf, Wsf, bsf, alpha)
    if newerror > error:
        return True
    else:
        return False
Winsf = 0.6
Wsf = 0.3
bsf = 0.5
print("bias test")
for i in range(10):
    print()
    value = run(Winsf, Wsf, bsf, 0.01)
    if value == True:
        bsf -= 0.05
    else:
        bsf += 0.05
alpha = 0.05
print("Alpha test")
for i in range(20):
    print()
    value = run(Winsf, Wsf, bsf, alpha)
    if value == True:
        alpha -= 0.01
    else:
        alpha += 0.01
