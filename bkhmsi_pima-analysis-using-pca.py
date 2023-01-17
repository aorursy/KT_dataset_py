# coding: utf-8

from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd 



from sklearn import preprocessing, decomposition

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.core import Dense, Activation, Dropout, Flatten

from keras.optimizers import SGD

from keras.utils import np_utils



np.random.seed(1337)  # for reproducibility
# Some global variables

num_classes = 1

num_features = 8

num_reduce = 7

epochs = 200

eig_vec = []
# You can ignore this, as this was part of my assignment

class PCA(object):

    def  __init__(self, k):

        self.U = None 

        self.mean = None

        self.std = None

        self.k = k



    def process(self, X_t):

        X = X_t.copy()

        pca_var = None

        if self.mean is None:

            self.mean = np.mean(X, axis=0)

            self.std = np.std(X, axis=0)



        X -= self.mean

        X /= self.std

        

        if self.U is None:

            cov = X.T.dot(X) / X.shape[0]

            self.U, S, V = np.linalg.svd(cov)

            pca_var = np.sum(S[:self.k]) / np.sum(S)

            

        return X.dot(self.U[:, :self.k]), pca_var       

        
def read_data():

    df = pd.read_csv("../input/diabetes.csv")

    data = df.as_matrix()

    y = data[:,  -1]

    X = data[:, :-1]

    return X, y
def get_model():

    model = Sequential()

    model.add(Dense(4,activation='elu',input_dim=(num_reduce)))

    model.add(Dense(6,activation='elu'))

    model.add(Dense(7,activation='elu'))

    model.add(Dense(8,activation='elu'))

    model.add(Dense(num_classes,activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])

    return model
def preprocess(X_train, X_val):

    X_t = X_train.copy()

    X_v = X_val.copy()

    mean = np.mean(X_t, axis=0)

    std = np.std(X_t, axis=0)

    X_t -= mean

    X_t /= std 

    X_v -= mean

    X_v /= std

    return X_t, X_v
def cross_val(X, y, k_fold = 5):

    step = X.shape[0] // k_fold 

    accuracies = []

    pca_kept_var = []



    for k in range(k_fold):

        # Divide dataset to training and validation sets

        X_val = X[k*step:((k+1)*step)]

        y_val = y[k*step:((k+1)*step)]

        X_train = np.delete(X,np.arange(k*step,((k+1)*step)), axis = 0)

        y_train = np.delete(y,np.arange(k*step,((k+1)*step)))



        if(num_features != num_reduce):

            pca = PCA(num_reduce)

            X_train, pca_var = pca.process(X_train)

            X_val, _ = pca.process(X_val)

            pca_kept_var.append(pca_var)

        else:

            X_train, X_val = preprocess(X_train, X_val)



        model = get_model()

        #model.summary() if k == 0 else None

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=50, validation_data=(X_val,y_val), verbose = 0)

        accuracies.append(np.max(history.history["val_acc"]))

        print("accuracy #",k,": ",accuracies[k])

    return np.mean(accuracies), np.mean(pca_kept_var) if len(pca_kept_var) > 0 else None
X, y = read_data()

print("X, Y shape", X.shape, y.shape)

acc, pca_var = cross_val(X, y)

print("ACCR: ", acc)

print("PCA Kept Variance: ", pca_var)