import json

import time



import joblib

import keras

import tensorflow

from keras.engine.saving import load_model

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D

from keras.regularizers import l2

from keras.optimizers import SGD, RMSprop

from keras.utils import to_categorical

from keras.layers.normalization import BatchNormalization

from keras.utils.vis_utils import plot_model

from keras.layers import Input, GlobalAveragePooling2D, concatenate, AveragePooling2D

from keras import models

from keras.models import Model

from keras_preprocessing.image import ImageDataGenerator

import numpy as np

from keras import regularizers

from keras import backend as K

from keras.applications.inception_v3 import InceptionV3

import matplotlib.pyplot as plt

#from keras.callbacks import tensorboard_v1
#train_root='../input/indoor-scenes-cvpr-2019/indoorCVPR_09/Images'#

train_root='../input/tinyscene/data/train'

#vaildation_root='/home/faith/keras/dataset/vaildationdata/'

test_root='../input/tinyscene/data/test'



LEARNING_RATE=0.01

MOMENTUM=0.9

ALPHA=0.0001

BETA=0.75

GAMMA=0.1

DROPOUT=0.4

WEIGHT_DECAY=0.0005

LRN2D_NORM=True

DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'

USE_BN=True



IM_WIDTH=299

IM_HEIGHT=299

EPOCH=200

batch_size=32

NB_CLASS=15



datagen = ImageDataGenerator(

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.3,

    rescale=1./255

)

train_generator = datagen.flow_from_directory(

  train_root,

  target_size=(IM_WIDTH, IM_HEIGHT),

  batch_size=batch_size,

  class_mode='categorical', subset='training'

)

#vaild data

vaild_generator = datagen.flow_from_directory(

  train_root,

  target_size=(IM_WIDTH, IM_HEIGHT),

  batch_size=batch_size,

  class_mode='categorical', subset='validation'

)

#test data

test_datagen = ImageDataGenerator(

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    featurewise_center=True

)

test_generator = datagen.flow_from_directory(

  test_root,

  target_size=(IM_WIDTH, IM_HEIGHT),

  batch_size=batch_size,

)
# create the base pre-trained model

base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 200 classes

predictions = Dense(NB_CLASS, activation='softmax')(x)

# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# train the model on the new data for a few epochs

model.fit_generator(train_generator,

                    validation_data=vaild_generator,

                    epochs=EPOCH,

                    steps_per_epoch=train_generator.n/batch_size,

                    validation_steps=vaild_generator.n/batch_size,

                    shuffle=True,

                    verbose=1)

model.save('scene_inception_3_1.h5')

history=model.history

joblib.dump((history), "scene_trainHistory1.pkl", compress=3)

# at this point, the top layers are well trained and we can start fine-tuning

# convolutional layers from inception V3. We will freeze the bottom N layers

# and train the remaining top layers.



# let's visualize layer names and layer indices to see how many layers

# we should freeze:

plot_model(model)

#model.summary()

plot_model(model, to_file='model.png')

'''for i, layer in enumerate(base_model.layers):

   print(i, layer.name)'''



# we chose to train the top 2 inception blocks, i.e. we will freeze

# the first 249 layers and unfreeze the rest:

for layer in model.layers[:17+1]:

   layer.trainable = False

for layer in model.layers[17+1:]:

   layer.trainable = True

# we need to recompile the model for these modifications to take effect

# we use SGD with a low learning rate

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])



# we train our model again (this time fine-tuning the top 2 inception blocks

# alongside the top Dense layers

model.fit_generator(train_generator,

                    validation_data=vaild_generator,

                    epochs=EPOCH,

                    steps_per_epoch=train_generator.n/batch_size,

                    validation_steps=vaild_generator.n/batch_size,

                    shuffle=True,

                    verbose=1)



model.save('scene_inception_3_2.h5')

history=model.history

joblib.dump((history), "scene_trainHistory2.pkl", compress=3)

loss,acc=model.evaluate_generator(test_generator,steps=vaild_generator.n/batch_size)

print('Test result:loss:%f,acc:%f'%(loss,acc))



# 绘制训练 & 验证的准确率值

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("scene_acc.png")

plt.show()



# 绘制训练 & 验证的损失值

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("scene_loss.png")

plt.show()
