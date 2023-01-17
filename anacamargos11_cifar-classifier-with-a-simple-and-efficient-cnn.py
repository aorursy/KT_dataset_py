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
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from __future__ import print_function

import numpy as np

import pandas as pd

from scipy import linalg

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import confusion_matrix

import keras

from tensorflow.keras import layers

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.utils import plot_model

import matplotlib.pyplot as plt
#------------------------------------------------------------------------------

# UNPACKING THE DATA FILES

#------------------------------------------------------------------------------ 

x_train = np.load('/kaggle/input/cifar10-comp/train_images.npy')

x_train = np.reshape(x_train,(50000,32,32,3))

y_train = pd.read_csv('/kaggle/input/cifar10-comp/train_labels.csv')

y_train = pd.Series(y_train['Category'])

# print the first 10 labels in y_train

y_train.head(10)
#------------------------------------------------------------------------------

# DATA PREPARATION

#------------------------------------------------------------------------------ 

lb=LabelBinarizer()

y_train=lb.fit_transform(y_train)

# Set aside the test and validation set

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2)

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.2)

# Standarisation

normalizer = keras.preprocessing.image.ImageDataGenerator(featurewise_center=True, 

    featurewise_std_normalization=True)

normalizer.fit(x_train)

normalizer.fit(x_val)
#------------------------------------------------------------------------------

# CREATING THE CNN

#------------------------------------------------------------------------------

nclasses = 10



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',activation='relu',input_shape=(32,32,3) ))

model.add(Conv2D(64, (3, 3), activation='relu') )

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu') )

model.add(Conv2D(256, (3, 3), activation='relu') )

model.add(MaxPooling2D(pool_size=(2, 2)) )

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))

model.add(Dropout(0.5))

model.add(Dense(nclasses, activation='softmax'))



opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
#------------------------------------------------------------------------------

# TRAINING THE CNN ON THE TRAIN/VALIDATION DATA

#------------------------------------------------------------------------------

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(x_train, y_train,

              batch_size=128,

              epochs=100,

              validation_data=(x_val, y_val),

              shuffle=True)
#------------------------------------------------------------------------------

# TESTING THE CNN ON THE TEST DATA

#------------------------------------------------------------------------------



# Normalizing test data

normalizer.fit(x_test)



# Score trained model.

scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])



pred = model.predict(x_test,verbose=1) 

pred_labels = lb.classes_[np.argmax(pred,axis=1)]   

labels = lb.classes_[np.argmax(y_test,axis=1)]  
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Check the model behavior for the first 10 images

for i in range(0,10): 

    print(pred_labels[i]==labels[i])

# Check the number of correct predictions

counter = 0

n_images = 10000

for i in range(0,n_images):

    cond = pred_labels[i]==labels[i]

    if cond:

        counter = counter + 1

print('Percentage of correct predictions:', counter/n_images)
# Compute the confusion matrix

cm = confusion_matrix(labels, pred_labels)

print(cm)