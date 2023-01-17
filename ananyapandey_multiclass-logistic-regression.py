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
import scipy.io as sc

data = sc.loadmat('../input/ex4data1.mat')
X = data['X']

y=data['y']

m,n = X.shape

m,n
X=X/255
# Lets visualize the data 

import matplotlib.pyplot as plt

(fig,axisarr) = plt.subplots(5,5,figsize=(5,5)) # returns a tuple so have to receive tuple i.e. (Something,Something)

for i in range(5):

    for j in range(5):

        temp=np.random.randint(X.shape[0]) # out of 5000 (shape of X ) generating a random number 

        axisarr[i,j].imshow(X[temp].reshape((20,20),order = 'F'))        

        axisarr[i,j].axis('off')

(fig,axx) = plt.subplots(1,5,figsize=(5,2))

for i in range(1,6):

    t=np.random.randint(5000)

    plt.subplot(1,5,i).imshow(X[t].reshape(20,20))

    #plt.subplot(1,5,i).imshow(X[temp].reshape(20,20))
def insert_ones(X):

    X_bias = np.c_[np.ones(X.shape[0]),X]

    return X_bias
def sigmoid(X):

    return 1/(1+np.exp(-1*X))



def h(theta,X):

    return np.dot(X,theta)
#def costfunction(theta,X,y): # without learning rate

#    m=len(y)

#    j = (-1/m)*np.sum( y * np.log(sigmoid(h(theta,X)))  + (1-y)*np.log(1-sigmoid(h(theta,X))))

#    return j



def costfunction(theta,X,y,lmbda=None): # lmbda as learning rate.

    m=len(y)

    pred=sigmoid(h(theta,X))

    pred[pred==1] = 0.9999

    j = (-1/m)*np.sum( y * np.log(pred)  + (1-y)*np.log(1-pred))

    return j+lmbda*np.sum( np.square(theta[1:]))/(2*m)
#def gradient(theta,X,y,lmbda=None): 

#    hypo = sigmoid(h(theta,X))

#    return np.dot(np.transpose(X),(hypo - y))
def gradientRegularized(theta,X,y,lmbda=None,alpha=1):

    m=len(y)

    gradtheta=theta

    gradtheta[0]=0

    return (alpha/m) * ( np.dot(np.transpose(X),(sigmoid(h(theta,X)-y))) + (lmbda*gradtheta) ) 
lmbda=0.1

num_labels=10 # for 10 classes 

#We are making 10 different models  one for each class. OneVsAll methoda

#  compute the “probability” that it belongs to each class using the trained logistic regression classifiers. 



def onevsall(X,y,num_labels,lmbda):

    m,n=X.shape

    theta=np.zeros(shape=(num_labels,n))

    for i in range(num_labels):

        num_label = i if i else 10

        import scipy.optimize as opt

        theta[i] = opt.fmin_cg(f=costfunction,x0=theta[i],fprime=gradientRegularized,args=(X,(y==num_label).flatten(),lmbda),maxiter=50)

        return theta

        
X=insert_ones(X)

theta=onevsall(X,y,num_labels,lmbda)
theta.shape

pred = np.argmax(sigmoid(X @ theta.T), axis = 1)

pred = [e if e else 10 for e in pred]

np.mean(pred == y.flatten()) *100