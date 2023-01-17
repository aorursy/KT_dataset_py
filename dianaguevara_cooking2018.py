# load python packages

import numpy as np

import scipy as sp

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import matplotlib

import IPython

import sklearn

import pprint

import json

from pprint import pprint



# import sklearn packages

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder





# import keras packages

import keras

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras import backend as K
# reading json

with open('../input/train.json') as f:

    trainJson = json.load(f)

    

with open('../input/test.json') as f:

    testJson = json.load(f)



trainTxt = [" ".join(doc['ingredients']).lower() for doc in trainJson]

testTxt = [" ".join(doc['ingredients']).lower() for doc in testJson]  



labelsTxt = [doc['cuisine'] for doc in trainJson]





# dimensionality transformations

vec = TfidfVectorizer(binary=True)

train = vec.fit_transform(trainTxt)

test = vec.transform(testTxt)  

enc = LabelEncoder()

label = enc.fit_transform(labelsTxt)



print(train, 'train samples')

print(test, 'test samples')

print(label, 'labels')
# type casting 

train = train.astype('float16')

test = test.astype('float16')

label = keras.utils.to_categorical(label)



print(train, 'train samples')

print(test, 'test samples')

print(label, 'labels')
# Model (3 layers with 1000 nodes)

model = keras.Sequential()

model.add(keras.layers.Dense(1000, 

                             kernel_initializer=keras.initializers.he_normal(seed=1), 

                             activation='relu', input_dim=3010))

model.add(keras.layers.Dropout(0.81))

model.add(keras.layers.Dense(1000, 

                             kernel_initializer=keras.initializers.he_normal(seed=2), 

                             activation='relu'))

model.add(keras.layers.Dropout(0.81))

model.add(keras.layers.Dense(20, 

                             kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=4), 

                             activation='softmax'))
# compile model

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# training

history = model.fit(train, label, 

                    epochs=30, 

                    batch_size=512, 

                    validation_split=0.1)

model.save_weights("model.h5")
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
predictions_encoded = model.predict(test)

predictions_encoded.shape

predictions = enc.inverse_transform([np.argmax(pred) for pred in predictions_encoded])

predictions
# print train and test losses and classification accuracies

score = model.evaluate(train, label, verbose=0)

print('Train loss:', score[0])

print('Train accuracy:', score[1])
Number_id = [doc['id'] for doc in testJson]

sub = pd.DataFrame({'id': Number_id, 'cuisine': predictions}, columns=['id', 'cuisine'])

sub.to_csv('output.csv', index=False)