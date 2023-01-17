import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import keras

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Embedding

from keras.layers import LSTM

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.layers import Input, Flatten, Dropout, Activation

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.models import Model

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras import regularizers

rnewdf = pd.read_pickle("../input/data.pkl")
rnewdf

newdf1 = np.random.rand(len(rnewdf)) < 0.8

train = rnewdf[newdf1]

test = rnewdf[~newdf1]



newdf1 = np.random.rand(len(rnewdf)) < 1
train[10:20]
trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]

testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder



X_train = np.array(trainfeatures)

y_train = np.array(trainlabel)

X_test = np.array(testfeatures)

y_test = np.array(testlabel)



lb = LabelEncoder()



y_train = np_utils.to_categorical(lb.fit_transform(y_train))

y_test = np_utils.to_categorical(lb.fit_transform(y_test))

x_traincnn =np.expand_dims(X_train, axis=2)

x_testcnn= np.expand_dims(X_test, axis=2)
model = Sequential()



model.add(Conv1D(256, 5,padding='same',input_shape=(216,1)))

model.add(Activation('relu'))

model.add(Conv1D(128, 5,padding='same'))

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(128, 5,padding='same',))

model.add(Activation('relu'))

model.add(Conv1D(128, 5,padding='same',))

model.add(Activation('relu'))

model.add(Conv1D(128, 5,padding='same',))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Conv1D(128, 5,padding='same',))

model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(10))

model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))
plt.plot(cnnhistory.history['loss'])

plt.plot(cnnhistory.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model_name = 'Emotion_Voice_Detection_Model.h5'

save_dir = os.path.join(os.getcwd(), 'saved_models')

# Save model and weights

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)
import json

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
# loading json and creating model

from keras.models import model_from_json

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")

print("Loaded model from disk")

 

# evaluate loaded model on test data

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
preds = loaded_model.predict(x_testcnn, 

                         batch_size=32, 

                         verbose=1)
preds1=preds.argmax(axis=1)
abc = preds1.astype(int).flatten()
predictions = (lb.inverse_transform((abc)))
preddf = pd.DataFrame({'predictedvalues': predictions})

preddf[:10]
actual=y_test.argmax(axis=1)

abc123 = actual.astype(int).flatten()

actualvalues = (lb.inverse_transform((abc123)))
actualdf = pd.DataFrame({'actualvalues': actualvalues})

actualdf[:10]
finaldf = actualdf.join(preddf)
finaldf[10:100]
finaldf.groupby('actualvalues').count()
finaldf.groupby('predictedvalues').count()