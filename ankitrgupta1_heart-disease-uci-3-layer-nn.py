# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def getData():

    heart_data = pd.read_csv("../input/heart.csv")

    X = heart_data.iloc[:,0:13]

    Y = heart_data.iloc[:,13]

    return X.values, Y.values
def scaleTrainData(train_X):

    scaler = StandardScaler()

    scaler.fit(train_X)

    scaled_train_X = scaler.transform(train_X)

    return scaler, scaled_train_X
def scaleTestData(scaler, test_X):

    scaled_test_X = scaler.transform(test_X)

    return scaled_test_X
def trainTestsplit(X,Y):

    train_X,test_X,train_y, test_y = train_test_split(X,Y, test_size=0.2, random_state=42)

    return train_X,test_X,train_y, test_y
def NNModel(scaled_train_X, train_Y):

    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(scaled_train_X.shape[0],), random_state=1, learning_rate_init=0.001, max_iter=200)

    clf.fit(scaled_train_X, train_Y)

    return clf
def getPrediction(clf, scaled_test_X):

    y_pred = clf.predict(scaled_test_X)

    return y_pred
def getPredictionMetrics(y_pred, y_actual):

    acc = "{0:.2f}".format(accuracy_score(y_actual, y_pred))

    precision = "{0:.2f}".format(precision_score(y_actual, y_pred))

    recall = "{0:.2f}".format(recall_score(y_actual, y_pred))

    f1 = "{0:.2f}".format(f1_score(y_actual, y_pred))

    return acc, precision, recall, f1
X,Y = getData()
train_X,test_X,train_Y, test_Y = trainTestsplit(X,Y)
scaler, scaled_train_X = scaleTrainData(train_X)
clf = NNModel(scaled_train_X, train_Y)
scaled_test_X = scaleTestData(scaler, test_X)
y_pred = getPrediction(clf, scaled_test_X)
acc, precision, recall, f1 = getPredictionMetrics(y_pred, test_Y)
print("accuracy = ", acc)

print("precision = ", precision)

print("recall = ", recall)

print("f1 = ", f1)