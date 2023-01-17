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
#accuracy: 0.9753 - val_loss: 0.3149 - val_accuracy: 0.8725


# Building a CNN

# Importing Keras Libraries and packages
from keras.models import Sequential       # To initialize the neural network
from keras.layers import Dense,Dropout    # To add fully connected layers
from keras.layers import Convolution2D    # To add convolution layers
from keras.layers import MaxPooling2D     # Maxpooling step
from keras.layers import Flatten          # To convert the features into single column
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Full architecture of CNN : Convolution => Max Pooling => Flattening => Full Connection

# Initialize the neural network
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation = 'relu')) 
classifier.add(Dropout(0.2))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))                   # It removes unnecessary nodes from the image but doesn't reduce the performance

# Adding a second convolutional layer
classifier.add(Convolution2D(64,(3,3) ,activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a third convolutional layer
classifier.add(Convolution2D(128,(3,3) ,activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Adding a forth convolutional layer
classifier.add(Convolution2D(256,(3,3) ,activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(output_dim=128 , activation = 'relu'))   # Hidden Layer
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim=1 , activation = 'sigmoid'))   # Output Layer

# Compiling the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

# Fitting the CNN to the Images

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                   rescale=1./255,
                                   rotation_range=20,
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True,
)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                '/kaggle/input/dataset/training_set',
                                                 target_size=(128,128),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            '/kaggle/input/dataset/test_set',
                                             target_size=(128,128),
                                             batch_size=32,
                                             class_mode='binary')


hst=classifier.fit_generator(
        training_set,
        steps_per_epoch=1000,                  #Steps_per_epoch = Total_samples/batch_size
        epochs=25,
        validation_data=test_set,
        validation_steps=200)                  #Validation_steps = Total_samples/batch_size






print(hst.history.keys())
# summarize history for accuracy
plt.plot(hst.history['accuracy'])
plt.plot(hst.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hst.history['loss'])
plt.plot(hst.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()