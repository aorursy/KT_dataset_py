# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
X,Y=make_blobs(n_samples=500,centers=2,n_features=2,random_state=3)
print(X.shape,Y.shape)
plt.style.use("seaborn")

plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.flag)

plt.xlim(-8,4)

plt.ylim(-4,8)

plt.show()
def sigmoid(z):

    return (1.0)/(1.0+(np.exp(-z)))



def predict(X,w):

    z=np.dot(X,w)

    predictions=sigmoid(z)

    return predictions



def loss(X,Y,w):

    #also called binary cross entropy

    Y_=predict(X,w)

    cost=np.mean((-Y*np.log(Y_)-(1-Y)*(np.log(1-Y_))))

    return cost



def update(X,Y,w,lr):

    #for one epoch

    Y_=predict(X,w)

    dw=np.dot(X.T,Y_-Y)

    m=X.shape[0]

    w=w-lr*dw/float(m)

    return w



def train(X,Y,lr=0.5,max_epochs=100):

    #modifing the input to handle the bias term

    ones=np.ones((X.shape[0],1))

    X=np.hstack((ones,X))

    

    # initiating with some random weights w

    

    w=np.zeros(X.shape[1])# n+1 entries

    

    #iterate overall epochs and make updates

    for epoch in range(max_epochs):

        w = update(X,Y,w,lr)

        

        if epoch % 10 == 0:

            l=loss(X,Y,w)

            

            print("epoch %d loss %.4f"%(epoch,l))

        

    return w
w = train(X,Y,lr=0.9,max_epochs=3500)

print(w)
def getPrediction(X_test,w,labels=True):

    if X_test.shape[1] != w.shape[0]:

        ones=np.ones((X_test.shape[0],1))

        X_test=np.hstack((ones,X_test))

        

    probs = predict(X_test,w)

    

    if not labels:

        return probs

    

    else:

        labels=np.zeros(probs.shape)

        labels[probs >=0.5] = 1

        

    return labels

x1=np.linspace(-8,2,10)

#generate equally distant 10 points ranging from -8 to 2

x2 = -(w[0]+w[1]*x1)/w[2]
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.flag)

plt.xlim(-8,4)

plt.ylim(-4,8)

plt.plot(x1,x2,c="blue")

plt.show()
Y_=getPrediction(X,w,labels=True)

training_acc=np.sum(Y_==Y)/Y.shape[0]



print(training_acc)