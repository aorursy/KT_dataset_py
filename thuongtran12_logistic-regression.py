# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import requests, gzip

from io import StringIO

import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import time



import seaborn as sns



# Any results you write to the current directory are saved as output.
# Load data

def load_mnist():

    """

    Arguments: None

    Returns:

            X_train - training data X

            Y_train - training label Y

            X_test  - testing data X

    """

    # Load the data

    train = pd.read_csv("../input/train.csv")

    test = pd.read_csv("../input/test.csv")

    

    y_train = train["label"]

    #Y_test  = train["label"]

    

    # Drop 'label' column

    X_train = train.drop(labels = ["label"], axis = 1)

    # Test

    X_test  = test

    

    # free some space

    del train

    

    # Histogram display

    #g = sns.countplot(Y_train)



    #Y_train.value_counts()

    

    # Normalize X

    X_train = X_train/255.0

    X_test  = X_test/255.0

    

    print(y_train)

    return X_train, y_train, X_test
# Visualize the data samples

X_train, y_train, X_test = load_mnist()

print('Dataset training size X is ',len(X_train))

print('Dataset testing size  X is ',len(X_test))

print('Dataset training size Y is ',len(y_train))



# Number of training

numtraining = len(X_train)



#Reshape X

X_train = X_train.values.reshape(-1,28,28)

X_test = X_test.values.reshape(-1,28,28)



print(X_train.shape)

plt.imshow(X_train[0]) 

plt.show()



print('Label: ', y_train[0])

y_train[1]
#------------------------------------------------------------#

# Machine learning model I - LINEAR REGRESSION

#------------------------------------------------------------#

scoring = ['precision', 'recall','f1','accuracy']



from sklearn import linear_model

from sklearn.metrics import accuracy_score

from sklearn.linear_model import SGDClassifier



# Reshape the data

trainXRG = X_train.reshape(-1,28*28)

trainY = y_train.reshape(-1,1)



def iter_minibatches(chunksize):

    # Provide chunks one by one

    chunkstartmaker = 0

    while chunkstartmaker < numtraining:

        chunkrows = range(chunkstartmaker, chunkstartmaker+chunksize)

        X_chunk = trainXRG[chunkrows] 

        y_chunk = trainY[chunkrows]

        yield X_chunk, y_chunk

        chunkstartmaker += chunksize



batcherator = iter_minibatches(chunksize=2000)

clfRG = SGDClassifier(max_iter=2000, n_jobs = 10, alpha = 0.001)



# Time record

tick = time.time()



# Train model

for X_chunk, y_chunk in batcherator:

    

    clfRG.partial_fit(X_chunk, y_chunk, np.unique(y_chunk))

    

    # Print time

    parsing_time = time.time() - tick

    print('Traing time: ', parsing_time)



# Now make predictions with trained model

y_predicted = clfRG.predict(trainXRG)



# Traing accuracy

accuracy_score(trainY, y_predicted)
# predict results

X_test = X_test.reshape(-1,28*28)

results = clfRG.predict(X_test)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("logistic_mnist_datagen.csv",index=False)