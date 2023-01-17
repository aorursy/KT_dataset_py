# Fix seeds

#from numpy.random import seed

#seed(1)

#from tensorflow import set_random_seed

#set_random_seed(2)



import numpy as np 

import pandas as pd

import os

from tqdm import tqdm

import time

from IPython import display

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



import tensorflow as tf

import keras

from keras import layers, models, activations

# Model



def build_neural_network(data_size_in, n_classes):

        

    inputs = layers.Input(shape=data_size_in)

    

    x = layers.Conv1D(kernel_size=5000, filters=8, activation='relu')(inputs)

    

    #x = layers.Conv1D(kernel_size=5000, filters=8, activation='relu')(x)

    

    x = layers.normalization.BatchNormalization()(x)

    

    x = layers.AveragePooling1D(pool_size=3)(x)

    

    x = layers.Conv1D(kernel_size=1000, filters=8, activation='relu')(x)

    

    #x = layers.Conv1D(kernel_size=1000, filters=8, activation='relu')(x)

    

    x = layers.normalization.BatchNormalization()(x)

    

    x = layers.AveragePooling1D(pool_size=3)(x)

    

    x = layers.Conv1D(kernel_size=500, filters=16, activation='relu')(x)

    

    #x = layers.Conv1D(kernel_size=500, filters=16, activation='relu')(x)

    

    x = layers.normalization.BatchNormalization()(x)

    

    x = layers.AveragePooling1D(pool_size=3)(x)

    

    x = layers.Conv1D(kernel_size=500, filters=16, activation='relu')(x)

    

    #x = layers.Conv1D(kernel_size=500, filters=16, activation='relu')(x)

    

    x = layers.normalization.BatchNormalization()(x)

    

    x = layers.AveragePooling1D(pool_size=3)(x)

    

    x = layers.Conv1D(kernel_size=150, filters=32, activation='relu')(x)

    

    #x = layers.Conv1D(kernel_size=150, filters=32, activation='relu')(x)

    

    x = layers.normalization.BatchNormalization()(x)

    

    x = layers.AveragePooling1D(pool_size=3)(x)

    

    x = layers.Conv1D(kernel_size=150, filters=64, activation='relu')(x)

    

    #x = layers.Conv1D(kernel_size=150, filters=64, activation='relu')(x)

    

    x = layers.normalization.BatchNormalization()(x)

    

    x = layers.MaxPool1D(pool_size=3)(x)

    

    x = layers.Flatten()(x)

        

    x = layers.Dense(units=500, activation='sigmoid')(x)

    

    x = layers.Dropout(rate=0.5)(x)

    

    x = layers.Dense(units=25, activation='relu')(x)

    

    predictions = layers.Dense(units=n_classes, activation='linear')(x)

    

    

    model = keras.models.Model(inputs=inputs, outputs=predictions)

    

    

    print(model.summary())

    return model
# load pretrained network



network_filepath = "../input/best-model/best_model.h5"

best_network=keras.models.load_model(network_filepath)
# Load submission file

submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})



# Load each test data, create the feature matrix, get numeric prediction

for i, seg_id in enumerate(tqdm(submission.index)):

  #  print(i)

    seg = pd.read_csv('../input/LANL-Earthquake-Prediction/test/' + seg_id + '.csv')

    x = seg['acoustic_data'].values

    temp = np.expand_dims(x,axis=0)

    submission.time_to_failure[i] = best_network.predict(np.expand_dims(temp, axis=2))



submission.head()



# Save

submission.to_csv('submission.csv')