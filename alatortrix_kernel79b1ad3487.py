# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.cluster import KMeans

from sklearn.metrics import r2_score as model_score, confusion_matrix

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.utils import to_categorical



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

animals = pd.read_csv("../input/train/train.csv")

test = pd.read_csv("../input/test/test.csv")

# Any results you write to the current directory are saved as output.

animals.head(3)
model_data = animals.drop(['AdoptionSpeed', 'Name', 'State', 'RescuerID', 'Description', 'PetID'], axis=1)

test_data = animals['AdoptionSpeed']

x_train, x_test, y_train, y_test = train_test_split(model_data, test_data, test_size=0.20, random_state=100)
model = Sequential()

model.add(Dense(5, input_dim=18))

model.add(Activation('softmax'))



model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



dummies = to_categorical(y_train, num_classes=5)

model.fit(x_train, dummies, epochs=10, batch_size=5)



score = model.evaluate(x_test, to_categorical(y_test, num_classes=5), batch_size=5)

score
pred = model.predict_classes(x_test)

confusion_matrix(pred, y_test)
model = Sequential()

model.add(Dense(5, input_dim=18))

model.add(Activation('relu'))

model.add(Dense(5, input_dim=18))

model.add(Activation('sigmoid'))



model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



dummies = to_categorical(y_train, num_classes=5)

model.fit(x_train, dummies, epochs=10, batch_size=5)



score = model.evaluate(x_test, to_categorical(y_test, num_classes=5), batch_size=5)

score
pred = model.predict_classes(x_test)

confusion_matrix(pred, y_test)
model = SVC(kernel='linear').fit(x_train,y_train)

pred = model.predict(x_test)

print("Modelo Lineal")

print("Acuraccy: ", metrics.accuracy_score(y_test,pred))

print(metrics.confusion_matrix(y_test,pred))  

print(metrics.classification_report(y_test,pred))
model = SVC(kernel='rbf').fit(x_train,y_train)

pred = model.predict(x_test)

print("Modelo Radial")

print("Acuraccy: ", metrics.accuracy_score(y_test,pred))

print(metrics.confusion_matrix(y_test,pred))  

print(metrics.classification_report(y_test,pred))
model = SVC(kernel='rbf', gamma='scale').fit(x_train,y_train)

pred = model.predict(x_test)

print("Modelo Radial")

print("Acuraccy: ", metrics.accuracy_score(y_test,pred))

print(metrics.confusion_matrix(y_test,pred))  

print(metrics.classification_report(y_test,pred))
model = SVC(kernel='sigmoid').fit(x_train,y_train)

pred = model.predict(x_test)

print("Modelo Sigmoidal")

print("Acuraccy: ", metrics.accuracy_score(y_test,pred))

print(metrics.confusion_matrix(y_test,pred))  

print(metrics.classification_report(y_test,pred))
model = SVC(kernel='sigmoid', gamma='scale').fit(x_train,y_train)

pred = model.predict(x_test)

print("Modelo Radial")

print("Acuraccy: ", metrics.accuracy_score(y_test,pred))

print(metrics.confusion_matrix(y_test,pred))  

print(metrics.classification_report(y_test,pred))