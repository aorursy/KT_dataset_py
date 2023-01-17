# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import time
df_train = pd.read_csv('../input/fashion-mnist_train.csv')
df_test = pd.read_csv('../input/fashion-mnist_test.csv')
print(df_train.shape)
print(df_test.shape)
df_train = np.array(df_train)
df_test = np.array(df_test)
df_train_y = df_train[:,0]
df_train_x = df_train[:,1:]
df_test_y = df_test[:,0]
df_test_x = df_test[:,1:]
print (df_train_x.shape)
print (df_train_y.shape)
print (df_test_x.shape)
print (df_test_y.shape)
x = np.linspace(0,df_train_y.shape[0],df_train_y.shape[0])
print(x.shape)
print(df_train_y.shape)
plt.scatter(x,df_train_y,c=df_train_y)
plt.show()
def drawImg(x):
    plt.imshow(x.reshape((28,28)),cmap='gray')
    plt.show()
def getDistance(x1,x2):
    x2 = np.reshape(x2,x1.shape)
    d = np.absolute(x1-x2)
    d = sum(d)
    return d
def predict(X_train,Y_train,X_test,k=5):
    dist = []
    for i in range(X_train.shape[0]):
        dist.append((getDistance(X_train[i],X_test),Y_train[i]))
    dist = np.array(dist)
    dist = dist[dist[:,0].argsort()]
    dist = dist[:k]
    dist = np.array(np.unique(dist[:,1],return_counts=True))
    dist = dist[0,np.argmax(dist[1,:])]
    return dist
print('Class Prediction for')
drawImg(df_test_x[:1])
print(predict(df_train_x,df_train_y,df_test_x[:1]))
print("Actual Value")
print(df_test_y[:1])
def getAccuracy(X_train,Y_train,X_test,Y_test,k=5):
    c = 0
    w = 0
    for i in range(X_test.shape[0]):
        p = predict(X_train,Y_train,X_test[i],k)
        if(p==Y_test[i]):
            c += 1
        else:
            w += 1
    acc = c/(c+w)
    acc = acc*100
    return c,w,acc
start = time.time()
cPred,wPred,acc = getAccuracy(df_train_x,df_train_y,df_test_x[:100],df_test_y[:100])
end = time.time()
print("Wrong Predictions are: ",wPred)
print("Correct Predictions are: ",cPred)
print("Accuracy is: ",acc,"%")
print("Total Time Elapsed: ",(end-start),"seconds")


