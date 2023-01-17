import numpy as np

import pandas as pd

import pydicom

import os

import collections

import sys

import glob

import random

import cv2

import tensorflow as tf

import multiprocessing



from math import ceil, floor

from copy import deepcopy

from tqdm import tqdm_notebook as tqdm

from imgaug import augmenters as iaa



import keras

import keras.backend as K

from keras.callbacks import Callback, ModelCheckpoint

from keras.layers import Dense, Flatten, Dropout

from keras.models import Model, load_model

from keras.utils import Sequence

from keras.losses import binary_crossentropy

from keras.optimizers import Adam
# Install Modules from internet

!pip install efficientnet

!pip install iterative-stratification
# Import Custom Modules

import efficientnet.keras as efn 

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
# Seed

SEED = 12345

np.random.seed(SEED)

tf.set_random_seed(SEED)



# Constants

TEST_SIZE = 0.02

HEIGHT = 256

WIDTH = 256

CHANNELS = 3

TRAIN_BATCH_SIZE = 32

VALID_BATCH_SIZE = 64

SHAPE = (HEIGHT, WIDTH, CHANNELS)



# Folders

DATA_DIR = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'

TEST_IMAGES_DIR = DATA_DIR + 'stage_2_test/'

TRAIN_IMAGES_DIR = DATA_DIR + 'stage_2_train/'
def correct_dcm(dcm):

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000



def window_image(dcm, window_center, window_width):    

    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):

        correct_dcm(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    

    # Resize

    img = cv2.resize(img, SHAPE[:2], interpolation = cv2.INTER_LINEAR)

   

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)

    return img



def bsb_window(dcm):

    brain_img = window_image(dcm, 40, 80)

    subdural_img = window_image(dcm, 80, 200)

    soft_img = window_image(dcm, 40, 380)

    

    brain_img = (brain_img - 0) / 80

    subdural_img = (subdural_img - (-20)) / 200

    soft_img = (soft_img - (-150)) / 380

    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return bsb_img



def _read(path, SHAPE):

    dcm = pydicom.dcmread(path)

    try:

        img = bsb_window(dcm)

    except:

        img = np.zeros(SHAPE)

    return img
# Image Augmentation

sometimes = lambda aug: iaa.Sometimes(0.25, aug)

augmentation = iaa.Sequential([ iaa.Fliplr(0.25),

                                iaa.Flipud(0.10),

                                sometimes(iaa.Crop(px=(0, 25), keep_size = True, sample_independently = False))   

                            ], random_order = True)       

        

# Generators

class TrainDataGenerator(keras.utils.Sequence):

    def __init__(self, dataset, labels, batch_size = 16, img_size = SHAPE, img_dir = TRAIN_IMAGES_DIR, augment = False, *args, **kwargs):

        self.dataset = dataset

        self.ids = dataset.index

        self.labels = labels

        self.batch_size = batch_size

        self.img_size = img_size

        self.img_dir = img_dir

        self.augment = augment

        self.on_epoch_end()



    def __len__(self):

        return int(ceil(len(self.ids) / self.batch_size))



    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        X, Y = self.__data_generation(indices)

        return X, Y



    def augmentor(self, image):

        augment_img = augmentation        

        image_aug = augment_img.augment_image(image)

        return image_aug



    def on_epoch_end(self):

        self.indices = np.arange(len(self.ids))

        np.random.shuffle(self.indices)



    def __data_generation(self, indices):

        X = np.empty((self.batch_size, *self.img_size))

        Y = np.empty((self.batch_size, 6), dtype=np.float32)

        

        for i, index in enumerate(indices):

            ID = self.ids[index]

            image = _read(self.img_dir+ID+".dcm", self.img_size)

            if self.augment:

                X[i,] = self.augmentor(image)

            else:

                X[i,] = image

            Y[i,] = self.labels.iloc[index].values        

        return X, Y

    

class TestDataGenerator(keras.utils.Sequence):

    def __init__(self, dataset, labels, batch_size = 16, img_size = SHAPE, img_dir = TEST_IMAGES_DIR, *args, **kwargs):

        self.dataset = dataset

        self.ids = dataset.index

        self.labels = labels

        self.batch_size = batch_size

        self.img_size = img_size

        self.img_dir = img_dir

        self.on_epoch_end()



    def __len__(self):

        return int(ceil(len(self.ids) / self.batch_size))



    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        X = self.__data_generation(indices)

        return X



    def on_epoch_end(self):

        self.indices = np.arange(len(self.ids))

    

    def __data_generation(self, indices):

        X = np.empty((self.batch_size, *self.img_size))

        

        for i, index in enumerate(indices):

            ID = self.ids[index]

            image = _read(self.img_dir+ID+".dcm", self.img_size)

            X[i,] = image              

        return X
def read_testset(filename = DATA_DIR + "stage_2_sample_submission.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df



def read_trainset(filename = DATA_DIR + "stage_2_train.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    duplicates_to_remove = [56346, 56347, 56348, 56349,

                            56350, 56351, 1171830, 1171831,

                            1171832, 1171833, 1171834, 1171835,

                            3705312, 3705313, 3705314, 3705315,

                            3705316, 3705317, 3842478, 3842479,

                            3842480, 3842481, 3842482, 3842483 ]

    df = df.drop(index = duplicates_to_remove)

    df = df.reset_index(drop = True)    

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df



# Read Train and Test Datasets

test_df = read_testset()

train_df = read_trainset()
# Oversampling

epidural_df = train_df[train_df.Label['epidural'] == 1]

train_oversample_df = pd.concat([train_df, epidural_df])

train_df = train_oversample_df



# Summary

print('Train Shape: {}'.format(train_df.shape))

print('Test Shape: {}'.format(test_df.shape))
def predictions(test_df, model):    

    test_preds = model.predict_generator(TestDataGenerator(test_df, None, 8, SHAPE, TEST_IMAGES_DIR), verbose = 1)

    return test_preds[:test_df.iloc[range(test_df.shape[0])].shape[0]]



def ModelCheckpointFull(model_name):

    return ModelCheckpoint(model_name, 

                            monitor = 'val_loss', 

                            verbose = 1, 

                            save_best_only = False, 

                            save_weights_only = True, 

                            mode = 'min', 

                            period = 1)



# Create Model

def create_model():

    K.clear_session()

    

    base_model =  efn.EfficientNetB2(weights = 'imagenet', include_top = False, pooling = 'avg', input_shape = SHAPE)

    x = base_model.output

    x = Dropout(0.15)(x)

    y_pred = Dense(6, activation = 'sigmoid')(x)



    return Model(inputs = base_model.input, outputs = y_pred)
# Submission Placeholder

submission_predictions = []



# Multi Label Stratified Split stuff...

msss = MultilabelStratifiedShuffleSplit(n_splits = 10, test_size = TEST_SIZE, random_state = SEED)

X = train_df.index

Y = train_df.Label.values



# Get train and test index

msss_splits = next(msss.split(X, Y))

train_idx = msss_splits[0]

valid_idx = msss_splits[1]
# Loop through Folds of Multi Label Stratified Split

#for epoch, msss_splits in zip(range(0, 9), msss.split(X, Y)): 

#    # Get train and test index

#    train_idx = msss_splits[0]

#    valid_idx = msss_splits[1]

for epoch in range(0, 8):

    print('=========== EPOCH {}'.format(epoch))



    # Shuffle Train data

    np.random.shuffle(train_idx)

    print(train_idx[:5])    

    print(valid_idx[:5])



    # Create Data Generators for Train and Valid

    data_generator_train = TrainDataGenerator(train_df.iloc[train_idx], 

                                                train_df.iloc[train_idx], 

                                                TRAIN_BATCH_SIZE, 

                                                SHAPE,

                                                augment = True)

    data_generator_val = TrainDataGenerator(train_df.iloc[valid_idx], 

                                            train_df.iloc[valid_idx], 

                                            VALID_BATCH_SIZE, 

                                            SHAPE,

                                            augment = False)



    # Create Model

    model = create_model()

    

    # Full Training Model

    for base_layer in model.layers[:-1]:

        base_layer.trainable = True

    TRAIN_STEPS = int(len(data_generator_train) / 6)

    LR = 0.000125



    if epoch != 0:

        # Load Model Weights

        model.load_weights('model.h5')    



    model.compile(optimizer = Adam(learning_rate = LR), 

                  loss = 'binary_crossentropy',

                  metrics = ['acc', tf.keras.metrics.AUC()])

    

    # Train Model

    model.fit_generator(generator = data_generator_train,

                        validation_data = data_generator_val,

                        steps_per_epoch = TRAIN_STEPS,

                        epochs = 1,

                        callbacks = [ModelCheckpointFull('efficientnet_model.h5')],

                        verbose = 1)

    

    # Starting with the 4th epoch we create predictions for the test set on each epoch

    if epoch >= 2:

        preds = predictions(test_df, model)

        submission_predictions.append(preds)
test_df.iloc[:, :] = np.average(submission_predictions, axis = 0, weights = [2**i for i in range(len(submission_predictions))])

test_df = test_df.stack().reset_index()

test_df.insert(loc = 0, column = 'ID', value = test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])

test_df = test_df.drop(["Image", "Diagnosis"], axis=1)

test_df.to_csv('efficientnet_submission.csv', index = False)

print(test_df.head(12))