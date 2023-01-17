# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import time

t0 = time.time()



import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



dataset = pd.read_csv("../input/kc_house_data.csv")



space=dataset['sqft_living']

price=dataset['price']



x = np.array(space).reshape(-1).astype(float)

y = np.array(price).astype(float)



def split(x,y,test_size):

    tot=len(x)

    t1=int(test_size*tot)

    arr = np.arange(tot)

    np.random.shuffle(arr)

    idtrain=arr[t1:]

    idtest=arr[:t1]

    print(len(idtrain),len(idtest),tot)

    print(idtrain,idtest)

    return x[idtrain],x[idtest],y[idtrain],y[idtest]



#Splitting the data into Train and Test

xtrain, xtest, ytrain, ytest = split(x,y,1/3)



def update(b,a,X,Y,learning_rate):

    db = da = 0

    tot=len(X)

    for i in range(tot):

        p=2*(b*X[i]+a-Y[i])

        db += p*X[i]

        da += p

    b -= db/float(tot) * learning_rate

    a -= da/float(tot) * learning_rate

    return b,a



def calc(x,y):

    return sum((y-x)**2)/len(x)

    print(x,y)



learning_rate = 0.0000001

b=a=1

for i in range(2000):

    b,a = update(b,a,xtrain,ytrain,learning_rate)

    #print(calc(x*b+a,y))

b=float(b)

a=float(a)

pred=xtrain*b+a

print(xtrain,pred)



print("h(x)=",b,"x+",a)

print("cost=",calc(x*b+a,y))

#Visualizing the training Test Results 

plt.scatter(xtrain, ytrain, color= 'red')

plt.plot(xtrain, xtrain*b+a, color = 'blue')

plt.title ("Visuals for Training Dataset")

plt.xlabel("Space")

plt.ylabel("Price")

plt.show()



#Visualizing the Test Results 

plt.scatter(xtest, ytest, color= 'red')

plt.plot(xtest, xtest*b+a, color = 'blue')

plt.title("Visuals for Test DataSet")

plt.xlabel("Space")

plt.ylabel("Price")

plt.show()

t1 = time.time()

print(t1-t0, "seconds wall time")