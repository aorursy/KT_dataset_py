import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.metrics import top_k_categorical_accuracy
from keras.utils import to_categorical
from keras.applications.nasnet import NASNetMobile
from os import listdir
from PIL import Image as PImage
from random import shuffle
import gc
from keras.preprocessing.image import img_to_array, array_to_img

files = list()

for num, a in enumerate(os.listdir('../input/fruits-360_dataset/fruits-360/Training/')):
    path = '../input/fruits-360_dataset/fruits-360/Training/' + a + '/'
    for file in os.listdir(path):
        files.append([path + file, num])

def loadImages(path, num):
    # return array of images
    y = list()
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        img = img.resize((224,224))
        loadedImages.append(img)
        y.append(num)
        
    return loadedImages, y

shuffle(files)
gc.collect()

def generator(bs = 64):
    for num, path in enumerate(files):
        if num % bs == 0:
            y = list()
            loadedImages = list()
        img = PImage.open(path[0])
        img = img.resize((224,224))
        loadedImages.append(img)
        y.append(path[1])
        if num % bs == bs-1:
            loadedImages = np.asarray([img_to_array(a) for a in loadedImages])
            y = np.asarray([a for a in y]).reshape(-1,1)
            yield (loadedImages, to_categorical(y, num_classes = 81))
           
files_valid = list()

for num, a in enumerate(os.listdir('../input/fruits-360_dataset/fruits-360/Test/')):
    path = '../input/fruits-360_dataset/fruits-360/Test/' + a + '/'
    for file in os.listdir(path):
        files_valid.append([path + file, num])

shuffle(files_valid)
        
def validation_generator(bs = 64):
    for num, path in enumerate(files_valid):
        if num % bs == 0:
            y = list()
            loadedImages = list()
        img = PImage.open(path[0])
        img = img.resize((224,224))
        loadedImages.append(img)
        y.append(path[1])
        if num % bs == bs-1:
            loadedImages = np.asarray([img_to_array(a) for a in loadedImages])
            y = np.asarray([a for a in y]).reshape(-1,1)
            yield (loadedImages, to_categorical(y, num_classes = 81))

import keras
from keras.models import Model
from keras.layers import Dropout, Input, Dense, Activation
model2 = keras.applications.nasnet.NASNetMobile(input_shape=(224, 224, 3), include_top=False, weights = 'imagenet', input_tensor = None, pooling='max')
inp = Input((224,224,3))
x = model2(inp)
x = Dense(81)(x)
x = Activation('softmax')(x)
model = Model(inp, x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
model.fit_generator(generator(bs=128), steps_per_epoch = 100, verbose = 1, epochs = 1)
res = model.evaluate_generator(validation_generator(), steps = 100, verbose=1)
print('Validaion Loss:', res[0], '. Accuracy:', res[1])