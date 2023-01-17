!pip install dbcollection
import dbcollection as dbc

import patoolib

import numpy as np

ucf=dbc.load('ucf')
X_train=ucf.get('train','images')

print(X_train.shape)

X_test=ucf.get("test",'images')

print(X_test.shape)

X_train=X_train.reshape(len(X_train),np.prod(X_train.shape[1:])).astype('float32')/255

X_test=X_test.reshape(len(X_test),np.prod(X_test.shape[1:])).astype('float32')/255

print("X_train",X_train.shape)

print("X_test",X_test.shape)