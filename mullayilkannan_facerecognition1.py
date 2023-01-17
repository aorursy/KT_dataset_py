# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import cv2

dataset_path = '../input/fer2013/fer2013.csv'

image_size=(48,48)

 

def load_fer2013():

    data = pd.read_csv(dataset_path)

    pixels = data['pixels'].tolist()

    width, height = 48, 48

    faces = []

    for pixel_sequence in pixels:

        face = [int(pixel) for pixel in pixel_sequence.split(' ')]

        face = np.asarray(face).reshape(width, height)

        face = cv2.resize(face.astype('uint8'),image_size)

        faces.append(face.astype('float32'))

    faces = np.asarray(faces)

    faces = np.expand_dims(faces, -1)

    emotions = pd.get_dummies(data['emotion']).as_matrix()

    return faces, emotions

 

def preprocess_input(x, v2=True):

    x = x.astype('float32')

    x = x / 255.0

    if v2:

        x = x - 0.5

        x = x * 2.0

    return x

 

faces, emotions = load_fer2013()

faces = preprocess_input(faces)

xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)

print(xtrain.shape) 
from matplotlib import pyplot as plt



for ix in range(10):

    print(ytrain[ix])

    plt.figure(ix)

    plt.imshow(xtrain[ix].reshape((48, 48)), interpolation='none', cmap='gray')

    plt.show()


from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

batch_size = 128

num_classes = 7

epochs = 12



 

print('xtrain shape:', xtrain.shape)

print(xtrain.shape[0], 'train samples')

print(xtest.shape[0], 'test samples')

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(48,48,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

#modified

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(48,48,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.fit(xtrain, ytrain,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(xtest, ytest))

score = model.evaluate(xtest, ytest, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

ypred = model.predict(xtest)

labels = np.argmax(ypred, axis=-1) 

print(labels)
 
BS=3

import pandas as pd 

from keras_preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,

     width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,

      horizontal_flip=True, fill_mode="nearest")

 

# train the network

H = model.fit_generator(aug.flow(xtrain, ytrain, batch_size=BS),

validation_data=(xtest, ytest), steps_per_epoch=len(xtrain) // BS,

epochs=3)
xtrain.shape

test2[0].shape

test2[0]


FACE_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EYE_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

SCALE_FACTOR = 1.3

BLUE_COLOR = (255, 0, 0)

MIN_NEIGHBORS = 5

frame=cv2.imread('../input/angrydata/angry2.jpg')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



faces = FACE_CLASSIFIER.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)

print (faces)

 


from keras.preprocessing.image import img_to_array

EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

#img_path='../input/testimage/'

#frame=cv2.imread('../input/testimage/mohanlalangry.jpg')

#reading the frame

#orig_frame = cv2.imread(img_path)

#frame = cv2.imread(img_path,0)

#faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

 

if len(faces) > 0:

    print(faces)

    faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

    (fX, fY, fW, fH) = faces

    print(fX, fY, fW, fH)

    roi = gray[fY:fY + fH, fX:fX + fW]

    roi = cv2.resize(roi, (48, 48))

    roi = roi.astype("float") / 255.0

    roi = img_to_array(roi)

    roi = np.expand_dims(roi, axis=0)

    preds = model.predict(roi)[0]

    emotion_probability = np.max(preds)

    label = EMOTIONS[preds.argmax()]

    #cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    #cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)

 

 
from matplotlib import pyplot as plt

print(label)

print(roi[0].shape)

plt.figure(1)

plt.imshow(roi[0].reshape(48, 48), interpolation='none', cmap='gray')

plt.show()
