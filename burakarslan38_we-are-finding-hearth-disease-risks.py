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
data_hearth = pd.read_csv("../input/heart.csv")

data_hearth.head(10)

data = data_hearth
data.info()
data.corr()
x,y = data_hearth.loc[:,data_hearth.columns != 'target'], data_hearth.loc[:,'target']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier

scores=[]

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    scores.append(knn2.score(x_test,y_test))

  

plt.plot(range(1,15),scores)

plt.show()


knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train,y_train)

predictions = knn.predict(x_test)

acc = knn.score(x_test,y_test)*100

print(acc)
from tensorflow import set_random_seed

set_random_seed(101)

import tensorflow

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 500)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))