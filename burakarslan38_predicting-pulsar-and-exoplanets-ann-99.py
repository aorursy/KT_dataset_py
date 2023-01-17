# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data_exo = pd.read_csv("../input/kepler-labelled-time-series-data/exoTrain.csv")

test_data_exo = pd.read_csv("../input/kepler-labelled-time-series-data/exoTest.csv")

data = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")

data.head()
data.corr()
train_data_exo.info()
test_data_exo.info()
X = data.drop(["target_class"],axis=1)

y = data["target_class"]
x_train_exo = train_data_exo.drop(["LABEL"],axis=1)

y_train_exo = train_data_exo["LABEL"]

x_test_exo = test_data_exo.drop(["LABEL"],axis=1)

y_test_exo = test_data_exo["LABEL"]
X_train,X_test, y_train , y_test = train_test_split(X,y,test_size=0.2)
lr = LogisticRegression()

lr.fit(X_train,y_train)

accuracy_lr=lr.score(X_test,y_test)

print("logistic regression accuracy:%{}".format(accuracy_lr*100))
from tensorflow import set_random_seed

set_random_seed(101)

import tensorflow

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train_exo.shape[1]))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 10)

accuracies = cross_val_score(estimator = classifier, X = x_train_exo, y = y_train_exo, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
print("Predicting Pulsar mean: "+ str(mean))

print("Predicting Pulsar variance: "+ str(variance))



print("Exoplanets accuracy:%{}".format(accuracy_lr*100))