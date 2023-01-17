from IPython.display import Image

Image("../input/inception/pokemons.png")
import numpy as np

import pandas as pd

import math



from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras import Sequential, Input

from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout, concatenate, AveragePooling2D

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import LearningRateScheduler
import cv2

import glob

import os
labels = np.array(list(os.walk('../input/pokemon-generation-one/dataset/'))[0][1])

np.random.shuffle(labels)
# Selecting only 10 pokemons to train

labels = labels[:10]
idx_to_name = {i:x for (i,x) in enumerate(labels)}

name_to_idx = {x:i for (i,x) in enumerate(labels)}
# Selected pokemons to train

idx_to_name
data = []

labels_one_hot = []



for label in labels:

    path = '../input/pokemon-generation-one/dataset/' + label + '/'

    imgs = np.array([cv2.resize(cv2.imread(img), (224,224), interpolation = cv2.INTER_AREA) for img in glob.glob(path + '*.jpg')])

    if(len(imgs) > 0):

        for i in imgs:

            labels_one_hot.append(name_to_idx[label])

        data.append(imgs)

    

data = np.array(data)

data = np.concatenate(data)



data = data / 255.0

data = data.astype('float32')

labels_one_hot = np.eye(len(list(idx_to_name.keys())))[labels_one_hot]
X_train, X_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.3, random_state=42)
kernel_init = tf.keras.initializers.glorot_uniform()

bias_init = tf.keras.initializers.Constant(value=0.2)
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce,

filters_5x5, filters_pool_proj, name=None):

    # create the 1x1 convolution layer that takes its input directly from the previous layer

    conv_1x1 = Conv2D(filters_1x1, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)



    # 3x3 route = 1x1 conv + 3x3 conv

    pre_conv_3x3 = Conv2D(filters_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pre_conv_3x3)



    # 5x5 route = 1x1 conv + 5x5 conv

    pre_conv_5x5 = Conv2D(filters_5x5_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_5x5 = Conv2D(filters_5x5, kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pre_conv_5x5)



    # pool route = pool layer + 1x1 conv

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)

    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)



    # concatenate the depth of the 3 filters together

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)



    return output
Image("../input/inception/inception.png")
# input layer with size = 24x24x3

input_layer = Input(shape=(224, 224, 3))

 

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)

 

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)

 

x = BatchNormalization()(x)

 

x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)

x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)

 

x = BatchNormalization()(x)

 

x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32, name='inception_3a')

 

x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64, name='inception_3b')

 

x = MaxPool2D((3, 3), padding='same', strides=(2, 2))(x)
x = inception_module(x, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64, name='inception_4a')

 

x = inception_module(x, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64, name='inception_4b')

 

x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64, name='inception_4c')

 

x = inception_module(x, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64, name='inception_4d')

 

x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128, name='inception_4e')

 

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
x = inception_module(x, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128, name='inception_5a')

 

x = inception_module(x, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128, name='inception_5b')
x = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(x)

x = Dropout(0.4)(x)

x = Flatten()(x)

x = Dense(10, activation='softmax', name='output')(x)
epochs = 500

initial_lrate = 0.01

 

# implement the learning rate decay function

def decay(epoch, steps=100):

    initial_lrate = 0.01

    drop = 0.96

    epochs_drop = 8

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate
lr_schedule = LearningRateScheduler(decay, verbose=0)

 

sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)
model = Model(input_layer, [x], name='googlenet')



model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=16, callbacks=[lr_schedule], verbose=0)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
print('Trying to predict ', labels[0])
img = cv2.imread('../input/pokemon-generation-one/dataset/' + labels[0] + '/' + list(os.walk('../input/pokemon-generation-one/dataset/' + labels[0] + '/'))[0][2][0])

img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)

img = img / 255.0

img = img.astype('float32')

img = np.array(img)

img = img.reshape(1, 224, 224, 3)
predicted = idx_to_name[np.argmax(model.predict(img))]

print('Predicted ', predicted)