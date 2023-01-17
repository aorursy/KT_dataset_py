import numpy as np
import pickle 
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, ceil
from keras.utils.np_utils import to_categorical

import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dropout, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, PReLU, Flatten, Dense, Input
from keras.layers import Layer, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from keras.engine.topology import Layer

"""
Dataset used is a preprocessed version of the German Traffic Signs Recognition Benchmark (GTSRB) dataset, details about which
can be found at [2]. This dataset is picked from [1]. The following code for reading data is borrowed from [3].
In particular data2.pickle is used, which performs shuffling and incorporates division by 255.0 followed by Mean Normalisation.
 

References
    [1] https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed
    [2] https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
    [3] https://www.kaggle.com/valentynsichkar/traffic-signs-classification-with-cnn
"""

# Opening file for reading in binary mode
with open('../input/traffic-signs-preprocessed/data2.pickle', 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # dictionary type

# Preparing y_train and y_validation for using in Keras
data['y_train'] = to_categorical(data['y_train'], num_classes=43)
data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)

# Making channels come at the end
data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

# Showing loaded data from file
for i, j in data.items():
    if i == 'labels':
        print(i + ':', len(j))
    else: 
        print(i + ':', j.shape)
          
"""
Inception Modules

An implementation of Google's Inception modules, introduced in GoogLeNet ([1]), done from scrath using Layer Subclassing.
The design of these implementations differs a bit from that of the original, and the deviations are inspired by [2].

References
    [1] Going deeper with convolutions, Christian Szegedy, Wei Liu, 2015
    [2] Traffic Sign Classification Using Deep Inception Based Convolutional Networks, Mrinal Hanoi, 2015

"""
class InceptionA(Layer):
    def __init__(self, ch_out_1, ch_out_3, ch_out_5, ch_out_m):
        super(InceptionA, self).__init__()
        self.conv1_1 = Conv2D(ch_out_1, 1, activation="relu")
        
        self.conv1_2 = Conv2D(ch_out_1, 1, activation="relu")
        self.conv3_1 = Conv2D(ch_out_3, 3, padding="same", activation=PReLU(), kernel_regularizer='l2')
        
        self.conv1_3 = Conv2D(ch_out_1, 1, activation="relu", padding="same")
        self.conv5_1 = Conv2D(ch_out_5, 5, padding="same", activation=PReLU(), kernel_regularizer='l2')
        
        self.conv1_4 = Conv2D(ch_out_1, 1, activation="relu")
        self.conv3_2 = Conv2D(ch_out_m, 3, padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.mp = MaxPool2D(pool_size=3, padding="same", strides=1)
        
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        
        y = self.conv1_2(inputs)
        y = self.conv3_1(y)
        
        z = self.conv1_3(inputs)
        z = self.conv5_1(z)
        
        w = self.conv1_4(inputs)
        w = self.conv3_2(w)
        w = self.mp(w)
        
        outputs = concatenate([x,y,z,w], axis=3)
        
        return outputs
    
    
class Inception4A(Layer):
    def __init__(self, ch_out_1, ch_out_3, ch_out_5, ch_out_m):
        super(Inception4A, self).__init__()
        self.conv1_1 = Conv2D(ch_out_1, 1, activation="relu")
        
        self.conv1_2 = Conv2D(ch_out_1, 1, activation="relu")
        self.conv3_1 = Conv2D(ch_out_3, 3, padding="same", activation=PReLU(), kernel_regularizer='l2')
        
        self.conv1_3 = Conv2D(ch_out_1, 1, activation="relu", padding="same")
        self.conv3_2 = Conv2D(ch_out_5, 3, padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.conv3_3 = Conv2D(ch_out_5, 3, padding="same", activation=PReLU(), kernel_regularizer='l2')
        
        self.conv1_4 = Conv2D(ch_out_1, 1, activation="relu")
        self.conv3_4 = Conv2D(ch_out_m, 3, padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.mp = MaxPool2D(pool_size=3, padding="same", strides=1)
        
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        
        y = self.conv1_2(inputs)
        y = self.conv3_1(y)
        
        z = self.conv1_3(inputs)
        z = self.conv3_2(z)
        z = self.conv3_3(z)
        
        w = self.conv1_4(inputs)
        w = self.conv3_4(w)
        w = self.mp(w)
        
        outputs = concatenate([x,y,z,w], axis=3)
        
        return outputs
    
class Inception4B(Layer):
    def __init__(self, ch_out_1, ch_out_3, ch_out_5, ch_out_m):
        super(Inception4B, self).__init__()
        self.conv1_1 = Conv2D(ch_out_1, 1, activation="relu")
        
        self.conv1_2 = Conv2D(ch_out_1, 1, activation="relu")
        self.conv7_1 = Conv2D(ch_out_3, (7,1), padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.conv7_2 = Conv2D(ch_out_3, (1,7), padding="same", activation=PReLU(), kernel_regularizer='l2')
        
        self.conv1_3 = Conv2D(ch_out_1, 1, activation="relu")
        self.conv7_3 = Conv2D(ch_out_5, (7,1), padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.conv7_4 = Conv2D(ch_out_5, (1,7), padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.conv7_5 = Conv2D(ch_out_5, (7,1), padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.conv7_6 = Conv2D(ch_out_5, (1,7), padding="same", activation=PReLU(), kernel_regularizer='l2')
        
        self.conv1_4 = Conv2D(ch_out_1, 1, activation="relu")
        self.conv3_1 = Conv2D(ch_out_m, 3, padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.mp = MaxPool2D(pool_size=3, padding="same", strides=1)
        
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        
        y = self.conv1_2(inputs)
        y = self.conv7_1(y)
        y = self.conv7_2(y)
        
        z = self.conv1_3(inputs)
        z = self.conv7_3(z)
        z = self.conv7_4(z)
        z = self.conv7_5(z)
        z = self.conv7_6(z)
        
        w = self.conv1_4(inputs)
        w = self.conv3_1(w)
        w = self.mp(w)
        
        outputs = concatenate([x,y,z,w], axis=3)
        
        return outputs
    
class Inception4C(Layer):
    def __init__(self, ch_out_1, ch_out_3, ch_out_5, ch_out_m):
        super(Inception4C, self).__init__()
        self.conv1_1 = Conv2D(ch_out_1, 1, activation="relu")
        
        self.conv1_2 = Conv2D(ch_out_1, 1, activation="relu")
        self.conv3_1 = Conv2D(ch_out_3/2, (1,3), padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.conv3_2 = Conv2D(ch_out_3/2, (3,1), padding="same", activation=PReLU(), kernel_regularizer='l2')
        
        self.conv1_3 = Conv2D(ch_out_1, 1, activation="relu", padding="same")
        self.conv3_3 = Conv2D(ch_out_5, (1,3), padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.conv3_4 = Conv2D(ch_out_5, (3,1), padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.conv3_5 = Conv2D(ch_out_5/2, (1,3), padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.conv3_6 = Conv2D(ch_out_5/2, (3,1), padding="same", activation=PReLU(), kernel_regularizer='l2')
        
        self.conv1_4 = Conv2D(ch_out_1, 1, activation="relu")
        self.conv3_7 = Conv2D(ch_out_m, 3, padding="same", activation=PReLU(), kernel_regularizer='l2')
        self.mp = MaxPool2D(pool_size=3, padding="same", strides=1)
        
        
    def call(self, inputs):
        x = self.conv1_1(inputs)
        
        y = self.conv1_2(inputs)
        y_1 = self.conv3_1(y)
        y_2 = self.conv3_2(y)
        y = concatenate([y_1, y_2], axis=3)
        
        z = self.conv1_3(inputs)
        z = self.conv3_3(z)
        z = self.conv3_4(z)
        z_1 = self.conv3_5(z)
        z_2 = self.conv3_6(z)
        z = concatenate([z_1, z_2], axis=3)
        
        w = self.conv1_4(inputs)
        w = self.conv3_7(w)
        w = self.mp(w)
        
        outputs = concatenate([x, y, z, w], axis=3)
        
        return outputs
"""
Spatial Transformer Network    
    
Introduced in [1], these networks learn an affine transformation to get the interesting part of the image into focus.
This implementation is borrowed from [2], and a lot was learnt prior to that from [3].
    
Here's an example usage of STN()

m = Sequential()
m.add(STN(input_shape=(16,16,3), filter_size=3))
m.add(Conv2D(64, 3, padding='same'))
m.add(MaxPool2D())
m.add(Conv2D(64, 3, padding='same'))
m.add(Dropout(0.5))
m.add(Flatten())
m.add(Dense(43, activation='softmax'))

m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
m.summary()

References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al., 2015
    [2]  https://github.com/mbanf/STN.keras
    [3]  https://github.com/hello2all/GTSRB_Keras_STN
"""

def K_meshgrid(x, y):
    return tf.meshgrid(x, y)

def K_linspace(start, stop, num):
    return tf.linspace(start, stop, num)

class BilinearInterpolation(Layer):

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super(BilinearInterpolation, self).__init__(**kwargs)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):

        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = .5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * K.cast(height, dtype='float32')

        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)

        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.flatten(base)

        base_y0 = y0 * width
        base_y0 = base + base_y0
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = K.reshape(image, shape=(-1, num_channels))
        flat_image = K.cast(flat_image, dtype='float32')
        pixel_values_a = K.gather(flat_image, indices_a)
        pixel_values_b = K.gather(flat_image, indices_b)
        pixel_values_c = K.gather(flat_image, indices_c)
        pixel_values_d = K.gather(flat_image, indices_d)

        x0 = K.cast(x0, 'float32')
        x1 = K.cast(x1, 'float32')
        y0 = K.cast(y0, 'float32')
        y1 = K.cast(y1, 'float32')

        area_a = K.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = K.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = K.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = K.expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = K_linspace(-1., 1., width)
        y_linspace = K_linspace(-1., 1., height)
        x_coordinates, y_coordinates = K_meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        ones = K.ones_like(x_coordinates)
        grid = K.concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return K.reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation, output_size):
        batch_size = K.shape(X)[0]
        transformations = K.reshape(affine_transformation, shape=(batch_size, 2, 3))
        regular_grids = self._make_regular_grids(batch_size, output_size[0], output_size[1])
        sampled_grids = K.batch_dot(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], output_size[2])
        interpolated_image = K.reshape(interpolated_image, new_shape)
        return interpolated_image
    
def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def STN(input_shape=(32, 32, 3), filter_size=5):
    image = Input(shape=input_shape)
    locnet = Conv2D(64, 5, strides=2, activation='relu')(image)
    locnet = MaxPool2D()(locnet)
    locnet = Conv2D(128, filter_size, activation='relu')(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(128)(locnet)
    weights = get_initial_weights(128)
    locnet = Dense(6, weights=weights)(locnet)
    out = BilinearInterpolation(input_shape)([image, locnet])
    return Model(inputs=image, outputs=out)

"""
A Deep Convolutional Neural Network

Model inspired by [1], but uses more recent Inception Modules. 
The presented network consists of spatial transformer layers and modified versions of inception modules specifically designed for capturing 
local and global features together. This features adoption allows the network to classify precisely intraclass samples even under deformations.
Use of spatial transformer layer makes this network more robust to deformations such as translation, rotation, scaling of input images.

References
    ---------
    [1] Traffic Sign Classification Using Deep Inception Based Convolutional Networks, Mrinal Hanoi, 2015

"""

input_shape = (32,32,3)

model = Sequential()
model.add(STN(input_shape))
model.add(Conv2D(64, 3, activation=PReLU(), padding='same'))
model.add(STN((32,32,64)))
model.add(Conv2D(64, 3, activation=PReLU(), padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D())

model.add(STN((16,16,64), 3))
model.add(Inception4A(32, 64, 16, 16))
model.add(STN((16,16,128), 3))
model.add(Inception4A(32, 64, 16, 16))
model.add(BatchNormalization())
model.add(MaxPool2D())

model.add(Inception4B(64, 128, 32, 32))
model.add(Inception4B(64, 128, 32, 32))
model.add(Inception4B(64, 128, 32, 32))
model.add(Inception4B(64, 128, 32, 32))
model.add(Inception4B(64, 128, 32, 32))
model.add(BatchNormalization())
model.add(MaxPool2D())

model.add(Inception4C(128, 256, 64, 64))
model.add(Inception4C(128, 256, 64, 64))
model.add(AvgPool2D(pool_size=4))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(43, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
"""
Training and Testing

An annealer is used to control lerning rate, reducing it as epochs increase so that fine learning can take place. 
Idea attributed to [1].

References 
    -----------
    [1] https://www.kaggle.com/valentynsichkar/traffic-signs-classification-with-cnn
"""

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
epochs = 30

h = model.fit(data['x_train'], data['y_train'],
              batch_size=20, epochs=epochs,
              validation_data=(data['x_validation'], data['y_validation']),
              callbacks=[annealer], verbose=1)

# Test Accuracy
pred = model.predict(data['x_test'])
corr = np.argmax(pred, axis=1)
acc = np.mean(corr == data['y_test'])
print("Test accuracy: ", acc)