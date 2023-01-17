# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import tensorflow as tf

from tensorflow import keras

import pandas as pd

import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

from matplotlib import rc

from sklearn.model_selection import train_test_split





%matplotlib inline



sns.set(style='whitegrid', palette='muted', font_scale=1.5)



rcParams['figure.figsize'] = 14, 8



RANDOM_SEED = 42



np.random.seed(RANDOM_SEED)
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv',header=0)

data.describe()

data.columns
data.info()

# Select the columns to use for prediction in the neural network

X= data.drop('target',axis=1)

Y=data['target']

print (X.shape, Y.shape, data.columns)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler



# split data into train, test

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=39, shuffle=True)

# normalize data

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)

X_test = pd.DataFrame(X_test_scaled)



print (X_train.shape, y_train.shape)

print (X_train.shape, y_test.shape)

print (data.columns)
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from keras.callbacks import ModelCheckpoint



model = Sequential()

model.add(Dense(30, input_dim=13, kernel_initializer='uniform', activation='relu'))

model.add(Dense(15, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="Adamax", metrics=['accuracy'])

model.summary()





# checkpoint

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')



callbacks_list = [checkpoint]

print ("trainning model....  please wait!")

history=model.fit(X_train, y_train, validation_split=0.33, epochs=100, batch_size=6, callbacks=callbacks_list,verbose=0 )



print ("model training - finished")
# load weights

model.load_weights("weights.best.hdf5")

# Compile model (required to make predictions)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Created model and loaded weights from file")

# load pima indians dataset

# split into input (X) and output (Y) variables



# estimate accuracy on whole dataset using loaded weights

core = model.evaluate(X_test, y_test, verbose=0)

print("score %s: %.2f%%" % (model.metrics_names[1], score[1]*100))



print("Evaluate model against trained data")

score = model.evaluate(X_train, y_train, verbose=0)

print("score %s: %.2f%%" % (model.metrics_names[1], score[1]*100))



print("Evaluate model against new data")

score = model.evaluate(X_test, y_test, verbose=0)

print("score %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print("Model prediction test")

# prediction return class type (1 or 0)

y_pred_class = model.predict_classes(X_test)

# prediction return proability percentage

y_pred_prob = model.predict(X_test)



print ("#  original | predicted  | probability  ")

for idx, label in enumerate(y_test):

    print ("%s     | %s  | %s |   %.2f%%" % (str(idx), str(label), str(y_pred_class[idx]), float(y_pred_prob[idx])*100))



# manually calculate accuracy rate

print("")

count = len(["ok" for idx, label in enumerate(y_test) if label == y_pred_class[idx]])

print ("Manually calculated accuracy is: %.2f%%" % ((float(count) / len(y_test))*100))

# using accuracy_score()

print ("Keras accuracy_score() is: %.2f%%" %  (accuracy_score(y_test, y_pred_class)*100))

print("")

print ("Simple confusion matrix ")

cm = confusion_matrix(y_test,y_pred_class)

print (cm)
# save trained model

trained_model_file="trained_heart_model.h5"

model.save_weights(trained_model_file)

print("Saved trained model to disk as h5 file :", trained_model_file)