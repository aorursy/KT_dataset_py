import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import tensorflow.keras as keras
import tensorflow as tf
import keras.backend as K
from keras.applications import Xception
from keras.layers import Dense, UpSampling2D, Conv2D, Activation, LeakyReLU, BatchNormalization
from keras import Model
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
base_model = Xception(weights=None, input_shape=(img_size, img_size, 3), include_top=False)
base_model.load_weights('../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

base_out = base_model.output
conv1 = Conv2D(1, (1, 1))(base_out)
conv1 = Activation('sigmoid')(conv1)
bn1 = BatchNormalization()(conv1)
re2 = LeakyReLU(0.2)(bn1)
up2 = UpSampling2D(16, interpolation='bilinear')(re2)
conv2 = Dense(264)(up2)
conv2 = Activation('softmax')(conv2)

model = Model(base_model.input, conv1)
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["categorical_accuracy"])
model.summary()

