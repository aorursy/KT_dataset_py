# dataset: https://www.kaggle.com/uciml/pima-indians-diabetes-database

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# flask api server 

# https://github.com/deepak6446/Spark-kafka-Real-Time-Analytics/blob/master/flask_api_server.py



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pickle



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load dataset and check shape

data = pd.read_csv("../input/diabetes.csv")

data.shape
# lets see some data

data.iloc[15:16]
# create json for prediction

x=dict()

for i in data.iloc[15:16].keys():

    x[i] = int(data.iloc[15:16][i])

print(x)
data.iloc[15:16].keys()
# lets create input output numpy array (drop lables from columns(1))

X = data.drop("Outcome", axis=1).values

Y = data["Outcome"].values

X.shape, Y.shape
# import sklearn's packages

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
# split train and test data with 70-30 ratio

# startify=Y makes sure that you split data evenly based on y(0, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=64, stratify=Y)
# create empty array for train and test accurracy

neighbour = np.arange(1,10)

train_accurracy = np.empty(len(neighbour))

test_accurracy = np.empty(len(neighbour))
for i, k in enumerate(neighbour):

    

    # algorithm='auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

    # n_jobs=-1 will use all available processer's

    knn = KNeighborsClassifier(algorithm="auto", n_jobs=-1, n_neighbors=k)

    knn.fit(X_train, Y_train)

    

    # accuracy on train data

    train_accurracy[i] = knn.score(X_train, Y_train)

    

    # accuracy on test data

    test_accurracy[i] = knn.score(X_test, Y_test)

    
# plot error function to see the optimal value of k for our dataset

plt.title("accuracy with varying k")

plt.plot(neighbour, train_accurracy, label = "train accuracy")

plt.plot(neighbour, test_accurracy, label = "test accuracy")

plt.legend()

plt.xlabel("number of neighbour")

plt.ylabel("accuracy")

plt.show()
# as we can see we get maximun accurracy for both test and train at k=5

# lets create a model with k=5

knn = KNeighborsClassifier(algorithm="auto", n_jobs=-1, n_neighbors=5)

knn.fit(X_train, Y_train)



# accuracy on train data

print("train accuracy", knn.score(X_train, Y_train))



# accuracy on test data

print("test accuracy", knn.score(X_test, Y_test))
pickle.dump(knn, open("model.sav", "wb"))

knn = pickle.load(open("model.sav", "rb"))

i=15

print("prediction for index:", i, "is ", knn.predict([X_test[i]]), "actual value", Y_train[i])