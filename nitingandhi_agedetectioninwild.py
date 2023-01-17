#Image preprocessing on AgeDB database (need to run only once)

'''

import cv2

import glob

import cvlib as cv

import numpy as np



cnt = 0

for img in glob.glob("/kaggle/input/agedb-database/AgeDB/*.jpg"):

    im = cv2.imread(img)

    im_cpy = np.copy(im)

    faces, confidences = cv.detect_face(im)

    for face in faces:    

        X, Y, Z = im.shape

        (startX,startY) = face[0],face[1]

        (endX,endY) = face[2],face[3]    

        if 0 < startX and X > endX and 0 < startY and Y > endY:

            cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2)

            startX2 = int(startX-(0.2*(endX-startX)))

            if startX2 < 0:

                startX2 = 0

            startY2 = int(startY-(0.2*(endY-startY)))

            if startY2 < 0:

                startY2 = 0

            endX2 = int(endX+(0.2*(endX-startX)))

            if endX2 > X:

                endX2 = X

            endY2 = int(endY+(0.2*(endY-startY)))

            if endY2 > Y:

                endY2 = Y

            cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2)

            cv2.rectangle(im, (startX2,startY2), (endX2,endY2), (0,0,255), 2)

    img_name = img[7:]

    img_name = img.split("/")[-1]

    try:

        #print("img_name: ", img_name)

        cv2.imwrite("images/marked_face/"+img_name, im)

        #print("images/marked_face/"+img_name)

        cropped_face = cv2.resize(im_cpy[startX2:endX2, startY2:endY2], (224, 224), interpolation = cv2.INTER_AREA)

        #cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)

    except:

        continue

    cv2.imwrite("images/cropped_face/"+img_name, cropped_face)

    #print("images/cropped_face/"+img_name)

    cnt += 1

    print("cnt: ", cnt)

print("Done!")

'''
#List all files utility

'''

import os

cwd = os.getcwd()

print(cwd)

cnt = 0

for root, dirs, files in os.walk(os.path.abspath(".")):

    for file in files:

        cnt += 1

        #print(os.path.join(root, file))

print("cnt: ", cnt)

'''
#Utility to save extracted faces (need to run only once)

'''

import os

import zipfile



def zipdir(path, ziph):

    # ziph is zipfile handle

    for root, dirs, files in os.walk("/kaggle/working/images/cropped_face"):

        for file in files:

            ziph.write(os.path.join(root, file))







cwd = os.getcwd()

print("cwd: ", cwd)

zipf = zipfile.ZipFile('marked.zip', 'w', zipfile.ZIP_DEFLATED)

zipdir('marked_face/', zipf)

zipf.close()

'''
import scipy.io

import numpy as np

import pandas as pd

from datetime import datetime, timedelta

import tensorflow as tf

import keras

from keras.preprocessing import image

from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation

from keras.layers import Conv2D, AveragePooling2D

from keras.models import Model, Sequential

from sklearn.model_selection import train_test_split

from keras import metrics

from keras.models import model_from_json

import matplotlib.pyplot as plt

import gc

import cv2

import os

import glob

import random



train_x = []

test_x = []

train_y = []

test_y = []



cwd = os.getcwd()

print("cwd: ", cwd)

#You can download our preprocessed dataset from here https://www.kaggle.com/nitingandhi/agedb-cropped

for img in glob.glob("/kaggle/input/agedb-cropped/cropped_face/*.jpg"):

  im = cv2.imread(img)

  im = cv2.resize(im, (224, 224), interpolation = cv2.INTER_AREA)

  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

  im = cv2.merge((im, im, im))

  y_tmp = int(img.split("_")[3])

  y_tmp_vec = np.zeros(101)

  y_tmp_vec[y_tmp-1] = 1  

  if random.random() < 0.94:

    train_x.append(im)

    train_y.append(y_tmp_vec)

  else:

    test_x.append(im)

    test_y.append(y_tmp_vec)



train_x = np.array(train_x)

train_y = np.array(train_y)

test_x = np.array(test_x)

test_y = np.array(test_y)



print("Train Count", train_x.shape, train_y.shape)

print("Test Count", test_x.shape, test_y.shape)

model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))

model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

 

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(128, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

 

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(256, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(256, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(256, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

 

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(512, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(512, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(512, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

 

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(512, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(512, (3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Convolution2D(512, (3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

 

model.add(Convolution2D(4096, (7, 7), activation='relu'))

model.add(Dropout(0.5))

model.add(Convolution2D(4096, (1, 1), activation='relu'))

model.add(Dropout(0.5))

model.add(Convolution2D(2622, (1, 1)))

model.add(Flatten())

model.add(Activation('softmax'))

#pre-trained weights of vgg-face model. 

#you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing

model.load_weights('/kaggle/input/vgg-face/vgg_face_weights.h5')
#freeze all layers of VGG-Face except last 7 one

for layer in model.layers[:-7]:

    layer.trainable = False



classes = 101 #(0, 100])

base_model_output = Sequential()

base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)

base_model_output = Flatten()(base_model_output)

base_model_output = Activation('softmax')(base_model_output)



age_model = Model(inputs=model.input, outputs=base_model_output)
sgd = keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)



age_model.compile(loss='categorical_crossentropy'

                  , optimizer=keras.optimizers.Adam()

                  #, optimizer = sgd

                  , metrics=['accuracy']

                 )
checkpointer = ModelCheckpoint(

    filepath='classification_age_model.hdf5'

    , monitor = "val_loss"

    , verbose=1

    , save_best_only=True

    , mode = 'auto'

)



scores = []
enableFit = False



if enableFit:

    epochs = 50

    batch_size = 256



    for i in range(epochs):

        print("epoch ",i)

        

        ix_train = np.random.choice(train_x.shape[0], size=batch_size)

        

        score = age_model.fit(

            train_x[ix_train], train_y[ix_train]

            , epochs=1

            , validation_data=(test_x, test_y)

            , callbacks=[checkpointer]

        )

        

        scores.append(score)

    

    #restore the best weights

    from keras.models import load_model

    age_model = load_model("classification_age_model.hdf5")

    

    age_model.save_weights('age_model_weights.h5')

        

else:

    #pre-trained weights for age prediction: https://drive.google.com/file/d/1YCox_4kJ-BYeXq27uUbasu--yz28zUMV/view?usp=sharing

    age_model.load_weights("/kaggle/input/age-model/age_model_weights.h5")
val_loss_change = []; loss_change = []

for i in range(0, len(scores)):

    val_loss_change.append(scores[i].history['val_loss'])

    loss_change.append(scores[i].history['loss'])



plt.plot(val_loss_change, label='val_loss')

plt.plot(loss_change, label='train_loss')

plt.legend(loc='upper right')

plt.show()
#loss and accuracy on validation set

age_model.evaluate(test_x, test_y, verbose=1)
predictions = age_model.predict(test_x)
output_indexes = np.array([i for i in range(0, 101)])

apparent_predictions = np.sum(predictions * output_indexes, axis = 1)
mae = 0

actual_mean = 0

for i in range(0 ,apparent_predictions.shape[0]):

    prediction = int(apparent_predictions[i])

    actual = np.argmax(test_y[i])

    

    abs_error = abs(prediction - actual)

    actual_mean = actual_mean + actual

    

    mae = mae + abs_error

    

mae = mae / apparent_predictions.shape[0]



print("mae: ",mae)

print("instances: ",apparent_predictions.shape[0])