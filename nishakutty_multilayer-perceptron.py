# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from matplotlib import pyplot
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
dataset =pd.read_csv("../input/diabetes.csv")
# split into input (X) and output (Y) variables
print(dataset.columns)
X = dataset.drop('Outcome',axis =1)
Y = dataset['Outcome']

def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=2)
lrate = LearningRateScheduler(step_decay)
callbacks_list = [early_stop,lrate]
history = model.fit(X_train, y_train, epochs=150,validation_data=(X_test,y_test), batch_size=10,  verbose=2,callbacks=callbacks_list)
pyplot.plot(history.history['loss'], color='blue')
pyplot.plot(history.history['val_loss'], color='orange')
# calculate predictions
predictions = model.predict(X_test)
print("predictions",predictions)
# round predictions
y_pred = [round(x[0]) for x in predictions]
y_pred = list(map(int, y_pred))
y_pred = pd.DataFrame(y_pred)
print('ypred',y_pred)
y_test = y_test.reset_index()
y_test.drop('index',axis =1,inplace = True)
from sklearn.metrics import accuracy_score
print("accuracy: ",accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

