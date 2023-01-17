import pandas as pd

import numpy as np

import os

from os import listdir

from glob import glob

import itertools

import fnmatch

import random

from PIL import Image

import zlib

import itertools

import csv

from tqdm import tqdm

import matplotlib.pylab as plt

import seaborn as sns

import cv2

import skimage

from skimage import transform

from skimage.transform import resize

import scipy

from scipy.misc import imresize, imread

from scipy import misc

import keras

from keras import backend as K

from keras import models, layers, optimizers

from keras.applications import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.models import Model, Sequential, model_from_json

from keras.layers import Dense, Dropout, Input, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Lambda, AveragePooling2D

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop

import sklearn

from sklearn import model_selection

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV

from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score

from sklearn.utils import class_weight

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
def loadBatchImages(path,nSamples,nVal):

    catList = listdir(path)

    loadedImagesTrain = []

    loadedLabelsTrain = []

    loadedImagesVal = []

    loadedLabelsVal = []

    for cat in catList[0:256]:

        deepPath = path+cat+"/"

        imageList = listdir(deepPath)

        indx = 0

        for images in imageList[0:nSamples + nVal]:                

            img = load_img(deepPath + images)

            img = img_to_array(img)

            img = misc.imresize(img, (224,224))

            if indx < nSamples:

                loadedLabelsTrain.append(int(images[0:3])-1)

                loadedImagesTrain.append(img)

            else:

                loadedLabelsVal.append(int(images[0:3])-1)

                loadedImagesVal.append(img)

            indx += 1

    return loadedImagesTrain, np_utils.to_categorical(loadedLabelsTrain), loadedImagesVal, np_utils.to_categorical(loadedLabelsVal) 



def shuffledSet(a, b):

    assert np.shape(a)[0] == np.shape(b)[0]

    p = np.random.permutation(np.shape(a)[0])

    return (a[p], b[p])



path = '/kaggle/input/caltech256-without-rename2/256_objectcategories/256_ObjectCategories/'

nSamples = 10

nVal = 7  

data, labels, dataVal, labelsVal = loadBatchImages(path,nSamples,nVal)

data = preprocess_input(np.float64(data))

dataVal = preprocess_input(np.float64(dataVal))

train = shuffledSet(np.asarray(data),labels)

val = shuffledSet(np.asarray(dataVal),labelsVal)

X_train = train[0]

y_train = train[1]

X_test = val[0]

y_test = val[1]
class_weight1 = None

weight_path1 = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

pretrained_model = VGG16(weights = weight_path1, include_top=False, input_shape=(224, 224, 3))

optimizer1 = keras.optimizers.RMSprop(lr=0.0001)



def vggModelWithNoTop(xtrain,ytrain,xtest,ytest,pretrainedmodel,pretrainedweights,classweight,numclasses,numepochs,optimizer,labels):

    base_model = pretrained_model 

    x = base_model.output

    x = Conv2D(256, kernel_size = (3,3), padding = 'valid')(x)

    x = Flatten()(x)

    x = Dropout(0.75)(x)

    predictions = Dense(numclasses, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:

        layer.trainable = False

    model.compile(loss='categorical_crossentropy', 

                  optimizer=optimizer, 

                  metrics=['accuracy'])

    model.summary()

    #Here I Train the Model

    history = model.fit(xtrain,ytrain, epochs=numepochs, validation_data=(xtest,ytest), verbose=2)

    #Here I Evaluate the model

    score = model.evaluate(xtest,ytest, verbose=0)

    print('\nTest loss  :', score[0])

    print('Test Accuracy:', score[1])

    y_pred = model.predict(xtest)

    return model

vggModelWithNoTop(X_train, y_train, X_test, y_test,pretrained_model,weight_path1,class_weight1,257,600,optimizer1,labels)