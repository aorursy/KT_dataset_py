import os
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from keras.models import Model,Sequential
from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D,Input,GlobalAveragePooling2D
from keras.layers import add,Activation,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.optimizers import SGD
%matplotlib inline

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)
image_width=150
image_height=150
batch_size=64

train_normal = os.listdir('../input/chest_xray/chest_xray/train/NORMAL/')
train_pneumonia = os.listdir('../input/chest_xray/chest_xray/train/PNEUMONIA/')

labels = 'Normal', 'Pneumonia'
sizes = [len(train_normal), len(train_pneumonia)]
colors = ['green', 'red']
explode = (0.1,0)
# Plot
plt.pie(sizes, labels=labels,
        colors=colors,
       explode=explode)
 
plt.axis('equal')
plt.show()
# borrowed from faizunnabi, thanks!
train_dir = Path("../input/chest_xray/chest_xray/train/")
test_dir =  Path("../input/chest_xray/chest_xray/test/")

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',
                                                 target_size=(image_width, image_height),
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('../input/chest_xray/chest_xray/test',
                                            target_size=(image_width, image_height),
                                            batch_size=batch_size,
                                            class_mode='binary')
def conv2d_unit(x, filters, kernels, strides=1):
    x = Conv2D(filters, kernels,
               padding='same',
               strides=strides,
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(inputs, filters):
    x = conv2d_unit(inputs, filters, (1, 1))
    x = conv2d_unit(x, 2 * filters, (3, 3))
    x = add([inputs, x])
    x = Activation('linear')(x)
    return x

def stack_residual_block(inputs, filters, n):
    x = residual_block(inputs, filters)
    for i in range(n - 1):
        x = residual_block(x, filters)
    return x
def darknet_base(inputs):
    x = conv2d_unit(inputs, 32, (3, 3))
    x = conv2d_unit(x, 64, (3, 3), strides=2)
    for _ in range(20):
        x = stack_residual_block(x, 32, n=1)
        x = conv2d_unit(x, 64, (3, 3), strides=2)
    return x
def darknet():
    inputs = Input(shape=(image_width, image_height, 3))
    x = darknet_base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='softmax')(x)
    model = Model(inputs, x)
    return model
model = darknet()
sgd = SGD(lr=0.015, 
          decay=1e-6, 
          momentum=0.8, 
          nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# thanks again to faizunnabi, for this snippet
num_epochs=50

history = model.fit_generator(training_set,
                    steps_per_epoch=5216//batch_size,
                    epochs=num_epochs,
                    validation_data=test_set,
                    validation_steps=624//batch_size)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(['train', 'test'])
plt.show()
loss, score = model.evaluate_generator(test_set)
print("Loss:     ", loss)
print("Accuracy: ", score)