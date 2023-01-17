# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#input data ............

dataset = pd.read_csv("../input/messidor_features.csv")

dataset.shape
X=dataset.iloc[:,0:19]

Y=dataset.iloc[:,19]

print(X)
#data preprocessing.....

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)



print(X.shape)
#split data.............

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

print(X_train.shape)

print( X_test.shape)
#import libraries....



from keras.datasets import mnist

from keras.models import Sequential, load_model

from keras.layers.core import Dense, Dropout, Activation
#ann...........



model = Sequential()

model.add(Dense(128, input_shape=(19,)))

model.add(Activation('relu'))                            



model.add(Dense(56))

model.add(Activation('relu'))



#final......

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.summary()
from keras import optimizers

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

#fit the model...



fit_model = model.fit(X_train, y_train,epochs=20)

print(model.metrics_names)

score=model.evaluate(X_test, y_test, verbose=2)

print(score)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

#knn.......................



from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=19)

neigh.fit(X_train, y_train) 



y_pred = neigh.predict(X_test)





print(y_pred)



cnf_matrix = confusion_matrix(y_test, y_pred)

print(cnf_matrix)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)


from sklearn import svm

classifier = svm.SVC(kernel='linear', C=0.01)

y_pred = classifier.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)