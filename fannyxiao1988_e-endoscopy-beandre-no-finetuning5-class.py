!pip install efficientnet

import datetime

starttime = datetime.datetime.now()



import os

import sys

import cv2

import shutil

import random

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import multiprocessing as mp

import matplotlib.pyplot as plt



from keras.activations import elu



from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from keras import backend as K

from keras.models import Model

from keras.utils import to_categorical

from keras import optimizers, applications

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler,ModelCheckpoint



#from keras import load_weights

from sklearn.metrics import classification_report

from imgaug import augmenters as iaa







def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed) 

seed = 2020

seed_everything(seed)





import sys



import datetime

starttime = datetime.datetime.now()



import keras

import efficientnet.keras as efn 

from keras.engine import Layer, InputSpec

from keras import backend as K



from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping

from keras.models import Sequential

from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras.optimizers import Adam

from sklearn.metrics import classification_report

from imgaug import augmenters as iaa



import os

import random

import numpy as np





# Model parameters

N_CLASSES=5



FACTOR = 4

BATCH_SIZE = 16

EPOCHS = 20

WARMUP_EPOCHS = 5

LEARNING_RATE = 1e-4 * FACTOR

WARMUP_LEARNING_RATE = 1e-3 * FACTOR

HEIGHT =300

WIDTH = 300

CHANNELS = 3

TTA_STEPS = 5

ES_PATIENCE = 10

RLROP_PATIENCE = 3

DECAY_DROP = 0.5

LR_WARMUP_EPOCHS_1st = 2

LR_WARMUP_EPOCHS_2nd = 5



train_data ='../input/eendoscopybeandre/E-dataset-3or5/5-class/train'

valid_data ='../input/eendoscopybeandre/E-dataset-3or5/5-class/validation'

test_data  ='../input/eendoscopybeandre/E-dataset-3or5/5-class/test'
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(

    [   iaa.Fliplr(0.5), # horizontal flips

        iaa.Flipud(0.5),# vertical flips

        #iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),

        #iaa.Affine(shear=(-16, 16)),

        #iaa.Affine(rotate=(-45, 45)),

        #iaa.Affine(scale=(0.2, 1.2)),

        #iaa.ContrastNormalization((0.5, 1.5)),

        sometimes(iaa.Crop(percent=(0, 0.3))),# random crops

    ], random_order=True 

)



def augment(img):

    seq_det = seq.to_deterministic()

    aug_image = seq_det.augment_image(img)

    return aug_image





datagen=ImageDataGenerator(rescale=1./255, 

                           rotation_range=360,

                           horizontal_flip=True,

                           vertical_flip=True,

                           #validation_split=0.15,

                           preprocessing_function=augment

                          )







train_generator = datagen.flow_from_directory(

    train_data,

    target_size=(HEIGHT, WIDTH),

    batch_size=BATCH_SIZE,

    color_mode='rgb',

    class_mode='categorical',

    #subset='training',

    shuffle=True,

    seed=seed

    )



valid_generator= datagen.flow_from_directory(

    valid_data,

    target_size=(HEIGHT, WIDTH),

    batch_size=BATCH_SIZE,

    color_mode='rgb',

    class_mode='categorical',

    #subset='validation',

    shuffle=True,

    seed=seed

    )



test_generator=datagen.flow_from_directory(

    test_data,

    target_size=(HEIGHT, WIDTH),

    batch_size=BATCH_SIZE,

    color_mode='rgb',

    class_mode='categorical',

    #subset='validation',

    shuffle=False,

    seed=seed

    )
STEP_SIZE = train_generator.n // BATCH_SIZE





import keras

#import efficientnet.keras as efn 

import efficientnet.keras as efn 

# Load in EfficientNetB5

effnet = efn.EfficientNetB3(weights='imagenet',

                        include_top=False,

                        input_shape=(HEIGHT, WIDTH, CHANNELS))

#effnet.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b5_imagenet_1000_notop.h5')



# Replace all Batch Normalization layers by Group Normalization layers

for i, layer in enumerate(effnet.layers):

    if "batch_normalization" in layer.name:

        effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)



def build_model():

    """

    A custom implementation of EfficientNetB5

    for the APTOS 2019 competition

    (Regression)

    """

    model = Sequential()

    model.add(effnet)

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))

    model.add(Dense(5, activation='softmax'))

    #model.add(Dense(1, activation="linear"))

    model.compile(loss='mse',

                  optimizer=Adam(lr=0.00005), 

                  metrics=['mse', 'acc'])

    print(model.summary())

    return model



# Initialize model

model = build_model()



# Monitor MSE to avoid overfitting and save best model

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=12)

rlr = ReduceLROnPlateau(monitor='val_loss', 

                        factor=0.5, 

                        patience=4, 

                        verbose=1, 

                        mode='auto', 

                        epsilon=0.0001)



# Begin training

model.fit_generator(train_generator,

                    steps_per_epoch=train_generator.samples // BATCH_SIZE,

                    epochs=35,

                    validation_data=valid_generator,

                    validation_steps = valid_generator.samples // BATCH_SIZE,

                    callbacks=[es, rlr])







model.save_weights('../working/effnetb3_fold0.h5')


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size



# Create empty arays to keep the predictions and labels

#lastFullTrainPred = np.empty((0, N_CLASSES))

#lastFullTrainLabels = np.empty((0, N_CLASSES))

lastFullValPred = np.empty((0, N_CLASSES))

lastFullValLabels = np.empty((0, N_CLASSES))





lastFulltestPred = np.empty((0, N_CLASSES))

lastFulltestLabels = np.empty((0, N_CLASSES))

'''

# Add train predictions and labels

for i in range(STEP_SIZE_TRAIN+1):

    im, lbl = next(train_generator)

    scores = model.predict(im, batch_size=train_generator.batch_size)

    lastFullTrainPred = np.append(lastFullTrainPred, scores, axis=0)

    lastFullTrainLabels = np.append(lastFullTrainLabels, lbl, axis=0)

'''





# Add validation predictions and labels

for i in range(STEP_SIZE_VALID+1):

    im, lbl = next(valid_generator)

    scores = model.predict(im, batch_size=valid_generator.batch_size)

    lastFullValPred = np.append(lastFullValPred, scores, axis=0)

    lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)

    

# Add validation predictions and labels

for i in range(STEP_SIZE_TEST+1):

    im, lbl = next(test_generator)

    scores = model.predict(im, batch_size=test_generator.batch_size)

    lastFulltestPred = np.append(lastFulltestPred, scores, axis=0)

    lastFulltestLabels = np.append(lastFulltestLabels, lbl, axis=0)

    

validation_preds = [np.argmax(pred) for pred in lastFullValPred]

validation_labels = [np.argmax(label) for label in lastFullValLabels]



test_preds = [np.argmax(pred) for pred in lastFulltestPred]

test_labels = [np.argmax(label) for label in lastFulltestLabels]



print(test_preds)

print(test_labels)

from sklearn.metrics import classification_report

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import confusion_matrix



print(confusion_matrix(test_labels, test_preds))



print(classification_report(validation_labels, validation_preds,digits=4))

print(classification_report(test_labels, test_preds,digits=4))



print("Train Cohen Kappa score: %.4f" % cohen_kappa_score(validation_labels, validation_preds, weights='quadratic'))

print("Train Cohen Kappa score: %.4f" % cohen_kappa_score(test_labels, test_preds, weights='quadratic'))