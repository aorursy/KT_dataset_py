import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import *

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ReduceLROnPlateau



import os

import random

print()
mySeed = 42



random.seed(mySeed)

np.random.seed(mySeed)

tf.set_random_seed(mySeed)
trainPath = '../input/dogs-vs-cats/train/train'

testPath = '../input/dogs-vs-cats/test1/test1'



test = os.listdir(testPath)

test = pd.DataFrame({'fileName': test})



dataset = os.listdir(trainPath)

dataset.sort()

xColumn, yColumn = 'fileName', 'label'

dataset = pd.DataFrame({xColumn: dataset})

def getClass(fileName):

    if 'dog' in fileName:

        className = 'dog'

    else:

        className = 'cat'

    return className



dataset[yColumn] = dataset[xColumn].apply(getClass)



train_x, test_x, train_y, test_y = train_test_split(dataset[xColumn].values, dataset[yColumn].values,

                                                   test_size = .25, random_state = mySeed)

trainset = pd.DataFrame({xColumn: train_x, yColumn:train_y})

valset = pd.DataFrame({xColumn: test_x, yColumn:test_y})



trainDataGenerator = ImageDataGenerator(rescale = 1./255,

                                       rotation_range = .1,

                                       width_shift_range = .1,

                                       height_shift_range = .1,

                                       brightness_range = (.9, 1),

                                       shear_range = .1,

                                       zoom_range = .1)



testDataGenerator = ImageDataGenerator(rescale = 1./255)



targetSize = [256, 256]

batchSize = 32

valBatchSize = 32



trainIterator = trainDataGenerator.flow_from_dataframe(dataframe = trainset,

                                                       directory = trainPath,

                                                       x_col = xColumn, y_col = yColumn,

                                                      target_size = targetSize,

                                                      batch_size = batchSize,

                                                      class_mode = 'binary')



valIterator = testDataGenerator.flow_from_dataframe(dataframe = valset,

                                                    directory = trainPath,

                                                    x_col = xColumn, y_col = yColumn,

                                                    target_size = targetSize,

                                                     batch_size = valBatchSize,

                                                     class_mode = 'binary')



testset = os.listdir(testPath)

testset = pd.DataFrame({xColumn: testset})

testIterator = testDataGenerator.flow_from_dataframe(dataframe = testset,

                                                    directory = testPath,

                                                    x_col = xColumn, y_col = None,

                                                    target_size = targetSize,

                                                     batch_size = valBatchSize,

                                                     class_mode = None,

                                                    shuffle = False)
inputShape = list(targetSize)

inputShape.append(3)



pretrained = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',input_shape=inputShape)

trainableLayers = 1

for layer in pretrained.layers[:18-trainableLayers]:

    layer.trainable = False

    pass

inputShape = pretrained.layers[-1].output_shape[1:]

layer = pretrained.layers[-1].output

layer = Conv2D(filters = 256, kernel_size = 3, activation = 'relu', padding='same')(layer)

layer = Conv2D(filters = 128, kernel_size = 1, activation = 'relu', padding='same')(layer)

layer = Conv2D(filters = 128, kernel_size = 3, activation = 'relu', padding='same')(layer)

layer = Conv2D(filters = 64, kernel_size = 1, activation = 'relu', padding='same')(layer)

layer = Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding='same')(layer)

layer = Conv2D(filters = 32, kernel_size = 1, activation = 'relu', padding='same')(layer)

layer = Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding='same')(layer)

layer = Conv2D(filters = 16, kernel_size = 1, activation = 'relu', padding='same')(layer)

layer = Conv2D(filters = 16, kernel_size = 3, activation = 'relu', padding='same')(layer)

layer = Conv2D(filters = 8, kernel_size = 1, activation = 'relu', padding='same')(layer)

kernalSize = int(layer.shape[1])

layer = Conv2D(filters = 1, kernel_size = kernalSize, activation = 'sigmoid')(layer)

output = Flatten()(layer)



classifier = Model(pretrained.input, output)



classifier.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
dirName, modelName = 'models', 'best.hd5'

if not os.path.exists(dirName):

    os.mkdir('models')

    pass

odelCheckpoint = ModelCheckpoint(filepath=os.path.join(dirName, modelName), monitor='val_acc', save_best_only=True)



reduceLROnPlateau = ReduceLROnPlateau(factor=0.3333333, patience=1, min_delta=0.0001, cooldown=1, min_lr=0.0001)



earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=7)



epochs = 29

classifier.fit_generator(trainIterator, epochs=epochs, callbacks=[odelCheckpoint, reduceLROnPlateau, earlyStopping], validation_data=valIterator)

classifier = load_model('models/best.hd5')
print(classifier.evaluate_generator(trainIterator))

print(classifier.evaluate_generator(valIterator))
epochs = 3

classifier.fit_generator(valIterator, epochs=epochs)
print(classifier.evaluate_generator(trainIterator))

print(classifier.evaluate_generator(valIterator))
labels = classifier.predict_generator(testIterator)
submit = pd.DataFrame({})

submit['id'] = testset.fileName.str.split('.').str[0]

submit['label']  = np.round(labels).astype(int)

submit.to_csv('submission_13010030.csv', index=False)
submit.head()