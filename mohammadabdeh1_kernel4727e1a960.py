# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# importing the required libraries to perform
# data Preprocessing training and testing and gragh ploting:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras import backend as K 
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from random import shuffle
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


TRAIN_DIR = '/kaggle/input/weed-detection-in-soybean-crops/dataset/train'
TEST_DIR = '/kaggle/input/weed-detection-in-soybean-crops/dataset/test'
DATASET_DIR = '/kaggle/input/weed-detection-in-soybean-crops/dataset'
IMG_SIZE = 180

label_list = {"soybean": 0,
              "soil": 1,
              "grass": 2,
              "broadleaf": 3}


import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't besaved outside of the current session

# creating the data array that will be used dor training and testing
# that is accessing the data folders, retriving images, and label them
# according to the dictionary above.
data = []
for label in label_list: 
    file_path = os.path.join(DATASET_DIR,label)
    for img in os.listdir(file_path):
        path = os.path.join(file_path, img)
        # reading the images 
        img = cv2.imread(path)
        # resizing the images to a common size
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        # appending images, along with their labels{0,1,2,3} to the data array
        data.append([np.array(img), label_list[label]])
# shuffling the array
shuffle(data)
np.save('train_data.npy', data)

# creating a list of training data 
X = np.array([i[0] for i in data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
# creating a list of labels 
Y = [i[1] for i in data]
# coding the labels in one-hot encoding; that is 0 -> {1,0,0,0}
# 1-> {0,1,0,0} etc..; this is the appropriate encoding to use 
# Softmax for classification 
nb_classes = 4
targets = np.array(Y)
one_hot_targets = np.eye(nb_classes)[targets]
# print(one_hot_targets)
# splitting the data into a training set and testing set.
X_train, X_test, Y_train, Y_test = train_test_split(X,one_hot_targets,test_size = 0.2, random_state = 0)

# splittin gthe training data into training set and validation set
# the validation set is used during the training to measure overfitting.
X_training, X_validating, Y_training, Y_validating = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)
print(X_training.shape)
print(X_validating.shape)
print(len(Y_training))
print(len(Y_validating))
# print(Y_validating)
# The designed CNN
model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(IMG_SIZE,IMG_SIZE,3), strides = 1, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (3,3), strides = 1, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (3,3), strides = 1, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512))
model.add(Dense(4, activation='softmax'))  # ACTION_SPACE_SIZE = how many choices (4)
model.compile(loss= tf.keras.losses.CategoricalCrossentropy(), optimizer=Adam(lr=0.00001), metrics=['categorical_accuracy'])
history = model.fit(X_training,np.array(Y_training), 
                    batch_size = 16, epochs=5, 
                    validation_data = (X_validating, Y_validating))
print(history.history)
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
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
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(X_test, Y_test, batch_size=1)
print('test loss, test acc:', results)
