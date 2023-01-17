import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import IPython

import warnings

warnings.filterwarnings("ignore")
y_test = np.random.randint(0,2,(3,4))

y_pred = np.random.random((3,4))

y_test,y_pred
def log_loss(y_test,y_pred):

    y_test = y_test.astype(np.float16)

    y_pred = y_pred.astype(np.float16)

    N,M = y_test.shape

    a=[]

    for m in range(M):

        loss=0

        for i in range(N):

            loss -= ((y_test[i,m]*np.log(y_pred[i,m]))+((1.0-y_test[i,m])*np.log(1.0-y_pred[i,m])))

        loss = loss/N

        a.append(round(loss,8))

    return a

a = log_loss(y_test,y_pred)

print(a)

np.mean(a)
tf.keras.losses.categorical_crossentropy(np.transpose(y_test), np.transpose(y_pred)).numpy()
print(tf.keras.losses.binary_crossentropy(np.transpose(y_test), np.transpose(y_pred)).numpy())

tf.keras.losses.binary_crossentropy(np.transpose(y_test), np.transpose(y_pred)).numpy().mean()
y_test = np.random.randint(0,2,(300,1))

y_pred = np.random.random((300,1))

a = log_loss(y_test,y_pred)

print(a)
y_pred[0.1>y_pred].shape,y_pred[y_pred>0.9].shape 
y = y_pred.clip(0.05,0.95)

log_loss(y_test,y)
y_test = np.random.randint(0,2,(300,3))

y_pred = np.random.random((300,3))

a = log_loss(y_test,y_pred)

print(a)
y = y_pred.clip(0.05,0.95)

log_loss(y_test,y)
a = np.array([1.0,0.0,1.0,0.0,1.0])

p = np.array([0.6,0.3,0.7,0.2,0.9])
tf.keras.losses.binary_crossentropy(a, p).numpy()
tf.keras.losses.binary_crossentropy(a, np.where(p,p>=0.5,1).astype(np.float16)).numpy()
a = np.array([1.0,0.0,1.0,0.0,1.0])

p = np.array([0.6,0.3,0.7,0.2,0.4])
tf.keras.losses.binary_crossentropy(a, p).numpy()
tf.keras.losses.binary_crossentropy(a, np.where(p,p>=0.5,1).astype(np.float16)).numpy()