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



!pip install '../input/kerasefficientnetb3/efficientnet-1.0.0-py3-none-any.whl' # Install EfficientNet

import efficientnet.keras as efn
HEIGHT = 137

WIDTH = 236

FACTOR = 0.60

HEIGHT_NEW = int(HEIGHT * FACTOR)

WIDTH_NEW = int(WIDTH * FACTOR)

CHANNELS = 3

BATCH_SIZE = 16



DIR = '../input/bengaliai-cv19'
# IMAGE PROCESSING

print(HEIGHT_NEW) # Image Size Summary

print(WIDTH_NEW)



def resize_image(img, WIDTH_NEW, HEIGHT_NEW):          # Image Prep

    img = 255 - img                                    # Invert

    img = (img * (255.0 / img.max())).astype(np.uint8) # Normalize

    img = img.reshape(HEIGHT, WIDTH) # Reshape

    image_resized = cv2.resize(img, (WIDTH_NEW, HEIGHT_NEW), interpolation = cv2.INTER_AREA)

    

    return image_resized.reshape(-1)   
# CREATE MODEL

# Generalized mean pool - GeM

gm_exp = tf.Variable(3.0, dtype = tf.float32)

def generalized_mean_pool_2d(X):

    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), axis = [1, 2], keepdims = False) + 1.e-7)**(1./gm_exp)

    return pool
# Create Model

def create_model(input_shape):

    input = Input(shape = input_shape) # Input Layer

    x_model = efn.EfficientNetB3(weights = None, include_top = False, input_tensor = input, pooling = None, classes = None) # Create and Compile Model and show Summary

    

    for layer in x_model.layers:   # UnFreeze all layers

        layer.trainable = True

    

    lambda_layer = Lambda(generalized_mean_pool_2d) # GeM

    lambda_layer.trainable_weights.extend([gm_exp])

    x = lambda_layer(x_model.output)

    

    grapheme_root = Dense(168, activation = 'softmax', name = 'root')(x) # multi output

    vowel_diacritic = Dense(11, activation = 'softmax', name = 'vowel')(x)

    consonant_diacritic = Dense(7, activation = 'softmax', name = 'consonant')(x)

   

    model = Model(inputs = x_model.input, outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])  # model



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

model2.load_weights('../input/kerasefficientnetb3/Train1_model_66.h5') # LB 0.9685

model3.load_weights('../input/kerasefficientnetb3/Train1_model_68.h5') # LB 0.9691

# model4.load_weights('../input/kerasefficientnetb3/Train1_model_57.h5') # LB ??

model5.load_weights('../input/kerasefficientnetb3/Train1_model_70.h5') # LB ??
# DATA GENERATOR

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
# PREDICT AND SUBMISSION



tgt_cols = ['grapheme_root','vowel_diacritic','consonant_diacritic'] # Create Submission File

row_ids, targets = [], [] # Create Predictions



for i in range(0, 4): # Loop through Test Parquet files (X)

    test_files = [] # Test Files Placeholder

    df = pd.read_parquet(os.path.join(DIR, 'test_image_data_'+str(i)+'.parquet')) # Read Parquet file

    image_ids = df['image_id'].values  # Get Image Id values

    df = df.drop(['image_id'], axis = 1) # Drop Image_id column

 

    X = [] # Loop over rows in Dataframe and generate images

    for image_id, index in zip(image_ids, range(df.shape[0])):

        test_files.append(image_id)

        X.append(resize_image(df.loc[df.index[index]].values, WIDTH_NEW, HEIGHT_NEW))

    

    data_generator_test = TestDataGenerator(X, batch_size = BATCH_SIZE, img_size = (HEIGHT_NEW, WIDTH_NEW, CHANNELS)) # Data_Generator

        

    # Predict with all 3 models

    preds1 = model1.predict_generator(data_generator_test, verbose = 1)

    preds2 = model2.predict_generator(data_generator_test, verbose = 1)

    preds3 = model3.predict_generator(data_generator_test, verbose = 1)

    preds4 = model4.predict_generator(data_generator_test, verbose = 1)

    preds5 = model5.predict_generator(data_generator_test, verbose = 1)

    

     

    for i, image_id in zip(range(len(test_files)), test_files): # Loop over Preds  

        for subi, col in zip(range(len(preds1)), tgt_cols):

            sub_preds1 = preds1[subi]

            sub_preds2 = preds2[subi]

            sub_preds3 = preds3[subi]

            sub_preds4 = preds4[subi]

            sub_preds5 = preds5[subi]



            row_ids.append(str(image_id)+'_'+col) # Set Prediction with average of 5 predictions

            sub_pred_value = np.argmax((sub_preds1[i] + sub_preds2[i] + sub_preds3[i] + sub_preds4[i] + sub_preds5[i]) / 5)

            targets.append(sub_pred_value)

   

    del df # Cleanup

    gc.collect()
# Create and Save Submission File

submit_df = pd.DataFrame({'row_id':row_ids,'target':targets}, columns = ['row_id','target'])

submit_df.to_csv('submission.csv', index = False)

print(submit_df.head(40))