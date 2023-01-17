from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.regularizers import l2

!pip install imutils

from imutils import paths

import matplotlib.pyplot as plt

import numpy as np

import argparse

import cv2

import os



from keras import applications



from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K
#Building a model with VGG19

class test_model:

    @staticmethod

    def build(width, height, depth, classes, reg, init="he_normal"):



        model = Sequential()

        inputShape = (height, width, depth)

        chanDim = -1





        if K.image_data_format() == "channels_first":

            inputShape = (depth, height, width)

            chanDim = 1



        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape = inputShape )

        for layer in base_model.layers:

            layer.trainable = False

        model = Sequential()

        model.add(base_model)

        # model.add(Dense(512, input_shape=(6, 6, 512), activation='relu'))

        model.add(Flatten())

        model.add(Dense(256, activation='relu'))

        model.add(Dense(256, activation='relu'))

        model.add(Dropout(0.5))

        model.add(Dense(classes, activation='softmax'))



        opt = Adam(lr=1e-4, decay=1e-4 / 50)

        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        model.summary()

        

        return model
#Loading the Data

#imagePaths = list(paths.list_images("/kaggle/input/101_objectcategories/101_ObjectCategories"))

path = "/kaggle/input/101_objectcategories/101_ObjectCategories"



categories = sorted(os.listdir(path))

ncategories = len(categories)



#print (imagePaths)



data_train = []

labels_train = []

data_test = []

labels_test = []



for i, category in enumerate(categories):

    counter = 0;

    for f in os.listdir(path + "/" + category):

        ext = os.path.splitext(f)[1]

        fullpath = os.path.join(path + "/" + category, f)

        #print(fullpath)

        label = fullpath.split(os.path.sep)[-2]

        image = cv2.imread(fullpath)

        image = cv2.resize(image, (144, 144))

        counter = counter + 1

        if (counter <= 30):      

            data_train.append(image)

            labels_train.append(label)

        else:

            data_test.append(image)

            labels_test.append(label)

            

print ('First 30 images are considered as Training and rest as testing')



data_train = np.array(data_train, dtype="float") / 255.0

data_test = np.array(data_test, dtype="float") / 255.0





lb = LabelBinarizer()



labels_train = lb.fit_transform(labels_train)

labels_test = lb.fit_transform(labels_test)



print(len(lb.classes_))
(trainX, testX1, trainY, testY1) = train_test_split(data_train, labels_train,

    test_size=0.0#, stratify=labels_train, random_state=42

                                                   )

(trainX2, testX, trainY2, testY) = train_test_split(data_test, labels_test,

    train_size=0.0#, stratify=labels_test, random_state=42

                                                   )



print("Data Splitted")



print(trainX.shape)

print(trainY.shape)

print(testX.shape)

print(testY.shape)
# Data Augmenting

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,

    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,

    horizontal_flip=True, fill_mode="nearest")



model = test_model.build(width=144, height=144, depth=3,classes=len(lb.classes_), reg=l2(0.0005))
# Evaluating the Model 



H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),

    validation_data=(testX, testY), steps_per_epoch=len(trainX) // 32,

    epochs=200,verbose=2)



scores = model.evaluate(testX, testY, verbose=2)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])