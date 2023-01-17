import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.utils import to_categorical





# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



# This is a bit of magic to make matplotlib figures appear inline in the notebook

# rather than in a new window.

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'
from sklearn.model_selection import train_test_split



# Import Data

train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

print("Train size:{}\nTest size:{}".format(train.shape, test.shape))



# Transform Train and Test into images\labels.

x_train = train.drop(['label'], axis=1).values.astype('float32') # all pixel values

y_train = train['label'].values.astype('int32') # only labels i.e targets digits

x_test = test.values.astype('float32')

#Reshape

x_train = x_train.reshape(x_train.shape[0], 28*28) / 255.0

x_test = x_test.reshape(x_test.shape[0], 28*28) / 255.0



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)



print(x_train.shape)

print(y_train.shape)

print(x_val.shape)

print(y_val.shape)

print(x_test.shape)
from sklearn import linear_model

linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)

print (f"linear's score:{linear.score(x_val,y_val)}")

linear.coef_         #coefficient

linear.intercept_    #intercept



print ("predict: ",linear.predict([x_test[0],x_test[1]]))
from sklearn import linear_model

logistic = linear_model.LogisticRegression()

logistic.fit(x_train,y_train)

print ("logistic's score: ",logistic.score(x_val,y_val))

logistic.coef_       #coefficient

logistic.intercept_  #intercept

print ("predict: ",logistic.predict([x_test[0],x_test[1]]))
from sklearn import tree

tree = tree.DecisionTreeClassifier(criterion='entropy')   # Gini、Information Gain、Chi-square、entropy

tree.fit(x_train,y_train)

print ("tree's score: ",tree.score(x_val,y_val))

print ("predict: ",tree.predict([x_test[0],x_test[1]]))
from sklearn import svm

svm = svm.SVC()

svm.fit(x_train,y_train)

print ("svm's score: ",svm.score(x_val,y_val))

print ("predict: ",svm.predict([x_test[0],x_test[1]]))
from sklearn import naive_bayes

bayes = naive_bayes.GaussianNB()

bayes.fit(x_train,y_train)

print ("bayes's score: ",bayes.score(x_val,y_val))

print ("predict: ",bayes.predict([x_test[0],x_test[1]]))
from sklearn import neighbors

KNN = neighbors.KNeighborsClassifier(n_neighbors = 3)

KNN.fit(x_train,y_train)

print ("KNN's score: ",KNN.score(x_val,y_val))

print ("predict: ",KNN.predict([x_test[0],x_test[1]]))