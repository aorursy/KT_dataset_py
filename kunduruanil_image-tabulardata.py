import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
!pip install efficientnet
from efficientnet import tfkeras as efn
train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
train.shape
size  = (256,256)
def get_model():
    model_input = tf.keras.Input(shape=(*size, 3), name='imgIn')
    tab_input = tf.keras.Input(shape=(3,),name="tabIn")
    dummy = tf.keras.layers.Lambda(lambda x:x)(model_input)
    outputs = []    
    for i in range(8):
        constructor = getattr(efn, f'EfficientNetB{i}')
 
        x = constructor(include_top=False, weights='imagenet', 
                        input_shape=(*size, 3), 
                        pooling='avg')(dummy)
 
        x = tf.keras.layers.Dense(100, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(50, activation='relu')(x)
        y = tf.keras.layers.Dense(100,activation="relu")(tab_input)
        y = tf.keras.layers.Dense(50,activation="relu")(y)
        concatenated = tf.keras.layers.concatenate([x, y], axis=-1)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(concatenated)
        outputs.append(output)
 
    model = tf.keras.Model([model_input,tab_input], outputs, name='aNetwork')
    model.compile(optimizer='adam',loss = tf.keras.losses.BinaryCrossentropy(
    label_smoothing = 0.05),metrics=[tf.keras.metrics.AUC(name='auc')])
    return model
model = get_model()
model.summary()
import sys
sys.getsizeof(model)
