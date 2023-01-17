import numpy as np
import pandas as pd

# imports needed for CNN
import csv
import cv2
import os, glob
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt

from subprocess import check_output
def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = os.listdir(data_dir)
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    category = 0
    for d in directories:
        if (d != 'Readme.txt'):
            file_names = [os.path.join(data_dir + d, f) for f in os.listdir(data_dir + d) if f.endswith(".ppm")]
        else:
            continue
        for f in file_names:
            img = cv2.imread(f)
            imresize = cv2.resize(img, (200, 125))
            #plt.imshow(imresize)
            images.append(imresize)
            labels.append(category) 
        category += 1
    #Normalization
    images = np.array(images).astype('float32')
    images = images / 255.0
    #hot encoding
    labels = np.array(labels)
    labels = to_categorical(labels,category)
    return images, labels
X_train, y_train = load_data('../input/BelgiumTSC_Training/Training/')
print (len(X_train))
print (len(y_train))
temp = -1
fig=plt.figure(figsize=(20, 20))
for i in range(0,y_train.shape[0]):
    if (np.argmax(y_train[i]) != temp):
        fig.add_subplot(10, 2, np.argmax(y_train[i]) + 1)
        plt.imshow(X_train[i])
        if (np.argmax(y_train[i]) >= 19):
            break
        temp = np.argmax(y_train[i])
X_test, y_test = load_data('../input/BelgiumTSC_Testing/Testing/')
print (len(X_test),len(y_test))
def createCNNModel(num_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(125, 200, 3), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    lrate = 0.01
    decay = lrate/30
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    return model
model = createCNNModel(y_train.shape[1])
model.fit(X_train, y_train, validation_split = 0.05, epochs=30)
predict = np.argmax(model.predict(X_test), axis = 1)
count = 0
for i in range(0,predict.shape[0]):
    if (predict[i] == np.argmax(y_test[i])):
        count +=1
print ('Accuracy on Test ',100 * count/predict.shape[0],'%')
model.predict(X_test[:100]).shape
