import cv2

import os

import time, gc

import numpy as np

import pandas as pd



import tensorflow as tf

import keras

from keras import backend as K

from keras.models import Model, Input

from keras.layers import Dense, Lambda

from math import ceil



# Install EfficientNet

!pip install '../input/kerasefficientnetb3/efficientnet-1.0.0-py3-none-any.whl'

import efficientnet.keras as efn
# Constants

HEIGHT = 137

WIDTH = 236

FACTOR = 0.70

HEIGHT_NEW = int(HEIGHT * FACTOR)

WIDTH_NEW = int(WIDTH * FACTOR)

CHANNELS = 3

BATCH_SIZE = 16



# Dir

DIR = '../input/bengaliai-cv19'
# Image Size Summary

print(HEIGHT_NEW)

print(WIDTH_NEW)



# Image Prep

def resize_image(img, WIDTH_NEW, HEIGHT_NEW):

    # Invert

    img = 255 - img



    # Normalize

    img = (img * (255.0 / img.max())).astype(np.uint8)



    # Reshape

    img = img.reshape(HEIGHT, WIDTH)

    image_resized = cv2.resize(img, (WIDTH_NEW, HEIGHT_NEW), interpolation = cv2.INTER_AREA)



    return image_resized.reshape(-1)   
# Generalized mean pool - GeM

gm_exp = tf.Variable(3.0, dtype = tf.float32)

def generalized_mean_pool_2d(X):

    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 

                        axis = [1, 2], 

                        keepdims = False) + 1.e-7)**(1./gm_exp)

    return pool
# Create Model

def create_model(input_shape):

    # Input Layer

    input = Input(shape = input_shape)

    

    # Create and Compile Model and show Summary

    x_model = efn.EfficientNetB3(weights = None, include_top = False, input_tensor = input, pooling = None, classes = None)

    

    # UnFreeze all layers

    for layer in x_model.layers:

        layer.trainable = True

    

    # GeM

    lambda_layer = Lambda(generalized_mean_pool_2d)

    lambda_layer.trainable_weights.extend([gm_exp])

    x = lambda_layer(x_model.output)

    

    # multi output

    grapheme_root = Dense(168, activation = 'softmax', name = 'root')(x)

    vowel_diacritic = Dense(11, activation = 'softmax', name = 'vowel')(x)

    consonant_diacritic = Dense(7, activation = 'softmax', name = 'consonant')(x)



    # model

    model = Model(inputs = x_model.input, outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])



    return model
# Create Model

model1 = create_model(input_shape = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))

model2 = create_model(input_shape = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))

model3 = create_model(input_shape = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))

model4 = create_model(input_shape = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))

model5 = create_model(input_shape = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))
# Load Model Weights

model1.load_weights('../input/kerasefficientnetb3/Train1_model_59.h5') # LB 0.9681

model2.load_weights('../input/kerasefficientnetb3/Train1_model_64.h5') # LB 0.9679

#model2.load_weights('../input/kerasefficientnetb3/Train1_model_66.h5') # LB 0.9685

model3.load_weights('../input/kerasefficientnetb3/Train1_model_68.h5') # LB 0.9691

model4.load_weights('../input/kerasefficientnetb3/Train1_model_57.h5') # LB ??

model5.load_weights('../input/kerasefficientnetb3/Train1_model_70.h5') # LB ??
class TestDataGenerator(keras.utils.Sequence):

    def __init__(self, X, batch_size = 16, img_size = (512, 512, 3), *args, **kwargs):

        self.X = X

        self.indices = np.arange(len(self.X))

        self.batch_size = batch_size

        self.img_size = img_size

                    

    def __len__(self):

        return int(ceil(len(self.X) / self.batch_size))



    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        X = self.__data_generation(indices)

        return X

    

    def __data_generation(self, indices):

        X = np.empty((self.batch_size, *self.img_size))

        

        for i, index in enumerate(indices):

            image = self.X[index]

            image = np.stack((image,)*CHANNELS, axis=-1)

            image = image.reshape(-1, HEIGHT_NEW, WIDTH_NEW, CHANNELS)

            

            X[i,] = image

        

        return X
# Create Submission File

tgt_cols = ['grapheme_root','vowel_diacritic','consonant_diacritic']



# Create Predictions

row_ids, targets = [], []



# Loop through Test Parquet files (X)

for i in range(0, 4):

    # Test Files Placeholder

    test_files = []



    # Read Parquet file

    df = pd.read_parquet(os.path.join(DIR, 'test_image_data_'+str(i)+'.parquet'))

    # Get Image Id values

    image_ids = df['image_id'].values 

    # Drop Image_id column

    df = df.drop(['image_id'], axis = 1)



    # Loop over rows in Dataframe and generate images 

    X = []

    for image_id, index in zip(image_ids, range(df.shape[0])):

        test_files.append(image_id)

        X.append(resize_image(df.loc[df.index[index]].values, WIDTH_NEW, HEIGHT_NEW))



    # Data_Generator

    data_generator_test = TestDataGenerator(X, batch_size = BATCH_SIZE, img_size = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))

        

    # Predict with all 3 models

    preds1 = model1.predict_generator(data_generator_test, verbose = 1)

    preds2 = model2.predict_generator(data_generator_test, verbose = 1)

    preds3 = model3.predict_generator(data_generator_test, verbose = 1)

    preds4 = model4.predict_generator(data_generator_test, verbose = 1)

    preds5 = model5.predict_generator(data_generator_test, verbose = 1)

    

    # Loop over Preds    

    for i, image_id in zip(range(len(test_files)), test_files):

        

        for subi, col in zip(range(len(preds1)), tgt_cols):

            sub_preds1 = preds1[subi]

            sub_preds2 = preds2[subi]

            sub_preds3 = preds3[subi]

            sub_preds4 = preds4[subi]

            sub_preds5 = preds5[subi]



            # Set Prediction with average of 5 predictions

            row_ids.append(str(image_id)+'_'+col)

            sub_pred_value = np.argmax((sub_preds1[i] + sub_preds2[i] + sub_preds3[i] + sub_preds4[i] + sub_preds5[i]) / 5)

            targets.append(sub_pred_value)

    

    # Cleanup

    del df

    gc.collect()
# Create and Save Submission File

submit_df = pd.DataFrame({'row_id':row_ids,'target':targets}, columns = ['row_id','target'])

submit_df.to_csv('submission.csv', index = False)

print(submit_df.head(40))