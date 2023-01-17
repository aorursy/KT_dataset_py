import numpy as np

import csv

import pandas

import matplotlib.pyplot as plt

from random import randint



import sys

sys.path.append(r"/kaggle/input/custom_module")

from logisticPak import *



path_train = '/kaggle/input/digit-recognizer/train.csv'

path_test = '/kaggle/input/digit-recognizer/test.csv'
trainset_num = 37000

raw = pandas.read_csv(path_train)

X = np.mat(raw.iloc[0:, 1:785]).T

y = np.mat(raw.iloc[0:, 0])



X = X*(1/255)

bias0 = np.ones((1,42000))

X = np.concatenate((bias0,X),axis=0)



theta = np.ones((10,785))



y_bi=(y==0)

for ybi in range(1,10):

    y_bi = np.concatenate((y_bi,(y==ybi)),axis=0)



X_valid = X[:,trainset_num: ]  

y_valid = y_bi[:, trainset_num:]

y_validReal = y[:, trainset_num:]

X = X[:,0:trainset_num]

y_bi = y_bi[:,0:trainset_num]



print('training set shape: ',X.shape,y_bi.shape)

print('cross validation set shape: ',X_valid.shape,y_valid.shape,y_validReal.shape)



iter_count = 0

iter2cost = np.mat([0, 1, 2, 3, 4])
rate = 1.5

iterTimes = 500  ##实际迭代次数为该参数的10倍



for cyc in range(0,iterTimes):

    for numi in range(0,10):

        for iter in range(0,10):

            grad_cur = gradCalc(trainset_num,X,y_bi[numi],theta[numi],0)

            theta[numi]=theta[numi]-rate*grad_cur



    cost_cur = 0

    costV_cur=0

    for numi in range(0,10):        

        cost_cur = cost_cur + costFunction(trainset_num,X,y_bi[numi],theta[numi],0)

        costV_cur = costV_cur + costFunction(42000-trainset_num,X_valid,y_valid[numi],theta[numi],0)

        

    pos_train = hypoFunction(X,theta)

    cla_train = pos_train.argmax(axis=0)

    accu_train=sum(cla_train==y[:,0:trainset_num])/trainset_num

        

    pos_valid = hypoFunction(X_valid,theta)

    cla_valid = pos_valid.argmax(axis=0)

    accu_valid=sum(cla_valid==y_validReal)/(42000-trainset_num)

        

    iter_count = iter_count+1

    i2c_cur = np.mat([iter_count,cost_cur[0,0],costV_cur[0,0], accu_train, accu_valid]) 

    iter2cost = np.concatenate((iter2cost, i2c_cur),axis=0)


fig = plt.figure(figsize=(10,12))

ax1 = fig.add_subplot(2,1,1)

plt.xlabel('itertations (*10)')

plt.ylabel('cost')

ax1.scatter(np.array(iter2cost[1:,0].T), np.array(iter2cost[1:,1].T), c='b',marker='o', label= 'training set')

ax1.scatter(np.array(iter2cost[1:,0].T), np.array(iter2cost[1:,2].T), c='r',marker='x',label= 'validation set')

ax1.legend()

plt.xlim(xmin=0)



ax2 = fig.add_subplot(2,1,2)

plt.ylabel('accuracy')

ax2.scatter(np.array(iter2cost[1:,0].T), np.array(iter2cost[1:,3].T), c='b',marker='o',label= 'training set')

ax2.scatter(np.array(iter2cost[1:,0].T), np.array(iter2cost[1:,4].T), c='r',marker='x',label= 'validation set')

ax2.legend()

plt.xlim(xmin=0)
print("final training set cost: %f" %(iter2cost[iterTimes,1]))

print("final cross validation cost: %f" %(iter2cost[iterTimes,2]))

print("accuracy of train data: %f" %(iter2cost[iterTimes,3]))

print("accuracy of cross validation data: %f" %(iter2cost[iterTimes,4]))
raw_test = pandas.read_csv(path_test)



X_t = np.mat(raw_test.iloc[:,0:]).T

X_t=X_t*(1/255)

bias0_t = np.ones((1,28000))

X_t = np.concatenate((bias0_t,X_t),axis=0)

print(X_t.shape)



pos_test = hypoFunction(X_t,theta)

cla_test = pos_test.argmax(axis=0)
start=0

cur_c = X_t[1:,start*10].reshape((28,28)) 

for itj in range(start*10+1,start*10+10): 

    add = X_t[1:,itj].reshape((28,28)) 

    cur_c = np.concatenate((cur_c,add),axis=1) 

    image_all = cur_c 

for iti in range((start+1),(start+10)): 

    cur_c = X_t[1:,iti*10].reshape((28,28)) 

    for itj in range(iti*10+1,iti*10+10): 

        add = X_t[1:,itj].reshape((28,28)) 

        cur_c = np.concatenate((cur_c,add),axis=1) 

    image_all= np.concatenate((image_all,cur_c),axis=0)



plt.imshow(image_all)

print(cla_test[...,start:start+100].reshape((10,10)))