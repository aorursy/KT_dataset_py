# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import warnings

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv'

data = pd.read_csv(path)
data.head()
data.columns
width, height = 48, 48

datapoints = data[' pixels'].tolist()



X = []

for xseq in datapoints:

    xx = [int(xp) for xp in xseq.split(' ')]

    xx = np.asarray(xx).reshape(width, height)

    X.append(xx.astype('float32'))



X = np.asarray(X)

X = np.expand_dims(X, -1)



#getting labels for training

y = pd.get_dummies(data['emotion']).as_matrix()



#storing them using numpy

#np.save('fdataX', X)

#np.save('flabels', y)



#print("Preprocessing Done")

#print("Number of Features: "+str(len(X[0])))

#print("Number of Labels: "+ str(len(y[0])))

#print("Number of examples in dataset:"+str(len(X)))

#print("X,y stored in fdataX.npy and flabels.npy respectively")
import sys, os

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model

from tensorflow.keras.models import model_from_json
num_features = 64

num_labels = 7

batch_size = 64

epochs = 100

width, height = 48, 48



#x = np.load('./fdataX.npy')

#y = np.load('./flabels.npy')



X -= np.mean(X, axis=0)

X /= np.std(X, axis=0)
#splitting into training, validation and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)



#saving the test samples to be used later

#np.save('modXtest', X_test)

#np.save('modytest', y_test)



#desinging the CNN

model = Sequential()



model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(2*2*2*num_features, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(2*2*num_features, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(2*num_features, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_labels, activation='softmax'))
model.compile(loss=categorical_crossentropy,

              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),

              metrics=['accuracy'])
history = model.fit(np.array(X_train), np.array(y_train),

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(np.array(X_valid), np.array(y_valid)),

          shuffle=True)
history.history.keys()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
#saving the  model to be used later

fer_json = model.to_json()

with open("fer.json", "w") as json_file:

    json_file.write(fer_json)

model.save_weights("fer.h5")

print("Saved model to disk")