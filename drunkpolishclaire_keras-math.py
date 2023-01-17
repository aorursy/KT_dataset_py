from keras.models import Sequential

from keras.layers import Dense, Dropout, CuDNNLSTM, Flatten

import numpy as np

import os

import cv2

import random

import pickle



def prepare(filepath):

    IMG_SIZE = 100  # 50 in txt-based

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



DATADIR = "/kaggle/input/dataset/sasankyadati-guns-dataset-0eb7329/SasankYadati-Guns-Dataset-0eb7329"

CATEGORIES = ["Non-Violent", "Violent"]

training_data = []



new_array = None

for category in CATEGORIES:  # do dogs and cats



    path = os.path.join(DATADIR,category)  # create path to dogs and cats

    class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat



    for img in os.listdir(path):  # iterate over each image per dogs and cats

        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array

        new_array = cv2.resize(img_array, (100, 100))  # resize to normalize data size

        training_data.append([new_array, class_num])  # add this to our training_data





random.shuffle(training_data)



X = []

y = []



for features,label in training_data:

    X.append(features)

    y.append(label)



X = np.array(X).reshape(-1, 100, 100)



model = Sequential()



model.add(CuDNNLSTM(256, input_shape=X.shape[1:], return_sequences=True))

model.add(Dropout(0.2))



model.add(CuDNNLSTM(256))

model.add(Dropout(0.1))



model.add(Dense(250, activation='relu'))



model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.fit(X, y, batch_size=32, epochs=100, validation_split=0.3)

model.save('model.h5')

model.load_weights('model.h5')
