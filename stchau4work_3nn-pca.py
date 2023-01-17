#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Sun Feb 12 21:37:36 2017



@author: steven

"""



import pandas as pd

from sklearn.decomposition import PCA

from sklearn.preprocessing import Normalizer

from sklearn.cross_validation import train_test_split

import numpy as np

from sklearn.neighbors import KNeighborsClassifier



# The competition datafiles are in the directory ../input

# Read competition data files:

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")



# Write to the log:

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Any files you write to the current directory get shown as outputs



def pca(X_tr, X_ts, teste, n):

    pca = PCA(n)

    pca.fit(X_tr)

    X_tr_pca = pca.transform(X_tr)

    X_ts_pca = pca.transform(X_test)

    teste = pca.transform(teste)    

    return X_tr_pca, X_ts_pca, teste



y = train[['label']]

X = train.drop('label',1)

header = X.columns



    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)



norm = Normalizer().fit(X_train)

X_train = norm.transform(X_train)

X_test = norm.transform(X_test)

test = norm.transform(test)



X_train = pd.DataFrame(X_train, columns = header)

X_test = pd.DataFrame(X_test, columns = header)

test = pd.DataFrame(test, columns = header)

y_train = pd.DataFrame(y_train)

y_train = y_train.as_matrix()



X_tr_pca, X_ts_pca, test = pca(X_train, X_test, test, 50)



X_tr_pca = pd.DataFrame(X_tr_pca) 

X_ts_pca = pd.DataFrame(X_ts_pca)

test = pd.DataFrame(test)



submission = pd.DataFrame()



    

model = KNeighborsClassifier(n_neighbors = 3, weights='distance')

model.fit(X_tr_pca, y_train)

score = model.score(X_ts_pca, y_test)

print ('KNN ', score)

pred = model.predict(test)

    

    



submission = pd.DataFrame({

    "ImageId": np.arange(1, pred.shape[0] + 1),

    "Label": pred

})



submission.to_csv("3nn_50pca.csv", index=False)