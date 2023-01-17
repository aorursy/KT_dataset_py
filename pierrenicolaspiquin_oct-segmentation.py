import os
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

# keras
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K

import tensorflow as tf
# Settings
input_path = os.path.join('..', 'input', '2015_boe_chiu', '2015_BOE_Chiu')
subject_path = [os.path.join(input_path, 'Subject_0{}.mat'.format(i)) for i in range(1, 10)] + [os.path.join(input_path, 'Subject_10.mat')]

data_indexes = [10, 15, 20, 25, 28, 30, 32, 35, 40, 45, 50]

width = 768
height = 496
mat = scipy.io.loadmat(subject_path[0])
img_tensor = mat['images']
manual_fluid_tensor_1 = mat['manualFluid1']

img_array = np.transpose(img_tensor, (2, 0, 1))
manual_fluid_array = np.transpose(manual_fluid_tensor_1, (2, 0, 1))
plt.imshow(img_array[25])
plt.imshow(manual_fluid_array[25])
def thresh(x):
    if x == 0:
        return 0
    else:
        return 1

thresh = np.vectorize(thresh, otypes=[np.float])

def create_dataset(paths):
    x = []
    y = []
    
    for path in paths:
        mat = scipy.io.loadmat(path)
        img_tensor = mat['images']
        fluid_tensor = mat['manualFluid1']
        
        img_array = np.transpose(img_tensor, (2, 0 ,1)) / 255
        fluid_array = np.transpose(fluid_tensor, (2, 0 ,1))
        fluid_array = thresh(fluid_array)
        
        for idx in data_indexes:
            x += [np.expand_dims(img_array[idx], 2)]
            y += [np.expand_dims(fluid_array[idx], 2)]
        
    return np.array(x), np.array(y)

x_train, y_train = create_dataset(subject_path[:9])
x_val, y_val = create_dataset(subject_path[9:])
# CNN model
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

inputs = Input((height, width, 1))

c1_1 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
c1_2 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1_1)
p1 = MaxPooling2D((2, 2))(c1_2)

c2_1 = Conv2D(12, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
c2_2 = Conv2D(12, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2_1)
p2 = MaxPooling2D((2, 2))(c2_2)

c3_1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
c3_2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3_1)
p3 = MaxPooling2D((2, 2))(c3_2)

c4_1 = Conv2D(24, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
c4_2 = Conv2D(24, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4_1)
p4 = MaxPooling2D((2, 2))(c4_2)

c5_1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
c5_2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5_1)

c6_t = Conv2DTranspose(24, (2, 2), strides=(2, 2), padding='same') (c5_2)
c6_c = concatenate([c6_t, c4_2])
c6_1 = Conv2D(24, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6_c)
c6_2 = Conv2D(24, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6_1)

c7_t = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c6_2)
c7_c = concatenate([c7_t, c3_2])
c7_1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7_c)
c7_2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7_1)

c8_t = Conv2DTranspose(12, (2, 2), strides=(2, 2), padding='same') (c7_2)
c8_c = concatenate([c8_t, c2_2])
c8_1 = Conv2D(12, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8_c)
c8_2 = Conv2D(12, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8_1)

c9_t = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8_2)
c9_c = concatenate([c9_t, c1_2])
c9_1 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9_c)
c9_2 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9_1)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9_2)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()
results = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=11, epochs=10)
pred = model.predict(x_val)
pred = (pred > 0.5).astype(np.uint8)
# prediction visualization
nrows = 6
fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 10))
for i in range(nrows):
    axes[i][0].imshow(np.reshape(x_val[2*i], (496, 768)), cmap='Greys_r')
    axes[i][1].imshow(np.reshape(y_val[2*i], (496, 768)), cmap='Greys_r')
    axes[i][2].imshow(np.reshape(pred[2*i], (496, 768)), cmap='Greys_r')

plt.show()