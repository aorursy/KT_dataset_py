# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

from PIL import Image

print(os.listdir("../input"))



import keras

from keras.models import Sequential

from keras.layers import Conv2D,Activation

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import SeparableConv2D

from keras.layers import BatchNormalization

from keras.layers import Dropout

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.callbacks import ModelCheckpoint

from keras import optimizers,regularizers

# Any results you write to the current directory are saved as output.
DATA_DIR = "../input/chest_xray/chest_xray"

TEST_NORMAL = "../input/chest_xray/chest_xray/test/NORMAL"

TRAIN_NORMAL = "../input/chest_xray/chest_xray/train/NORMAL"

VALIDATION_NORMAl = "../input/chest_xray/chest_xray/val/NORMAL"

TEST_PNEUMONIA = "../input/chest_xray/chest_xray/test/PNEUMONIA"

TRAIN_PNEUMONIA = "../input/chest_xray/chest_xray/train/PNEUMONIA"

VALIDATION_NORMAl = "../input/chest_xray/chest_xray/val/PNEUMONIA"

os.listdir(DATA_DIR)
import random

rand_normal = random.choice(os.listdir(TRAIN_NORMAL))

rand_pnemonia = random.choice(os.listdir(TRAIN_PNEUMONIA))



nor_im = Image.open(TRAIN_NORMAL + "/"+rand_normal)

pne_im = Image.open(TRAIN_PNEUMONIA + "/"+rand_pnemonia)



a = plt.figure(figsize=(20,10))



a1 = a.add_subplot(1,2,1)

imgplot = plt.imshow(nor_im)

a1.set_title("normal")



a2 = a.add_subplot(1,2,2)

imgplot = plt.imshow(pne_im)

a2.set_title("pneumonia")
adama = optimizers.adam(lr=0.001)
train_gen = ImageDataGenerator(shear_range=0.2,horizontal_flip=True,rescale=1.0/255)

test_gen = ImageDataGenerator(rescale=1.0/255)

train_set = train_gen.flow_from_directory(directory="../input/chest_xray/chest_xray/train",

                                                  target_size = (128,128),

                                                  batch_size = 32,

                                                  class_mode="binary")

test_set = test_gen.flow_from_directory(directory="../input/chest_xray/chest_xray/test",

                                                  target_size = (128,128),

                                                  batch_size = 32,

                                                  class_mode="binary")

val_set = test_gen.flow_from_directory(directory="../input/chest_xray/chest_xray/val",

                                                  target_size = (128,128),

                                                  batch_size = 32,

                                                  class_mode="binary")
model = Sequential()



model.add(Conv2D(32,(3,3),activation="relu", input_shape=(128,128,3),name="conv_1.1"))

model.add(Conv2D(32,(3,3),activation="relu",name="conv_1.2"))

model.add(MaxPooling2D((2,2),name="pool_1"))



model.add(SeparableConv2D(64,(3,3),activation="relu",name="conv_2.1"))

model.add(SeparableConv2D(64,(3,3),activation="relu",name="conv_2.2"))

model.add(MaxPooling2D((2,2),name="pool_2"))



model.add(SeparableConv2D(64,(3,3),name="conv_3.1"))

model.add(BatchNormalization(name='bn_3.1'))

model.add(Activation("relu"))

model.add(SeparableConv2D(64,(3,3),activation="relu",name="conv_3.2"))

model.add(MaxPooling2D((2,2),name="pool_3"))



model.add(SeparableConv2D(128,(3,3),name="conv_4.1"))

model.add(BatchNormalization(name='bn_4.1'))

model.add(Activation("relu"))

model.add(SeparableConv2D(128,(3,3),activation="relu",name="conv_4.2"))

model.add(MaxPooling2D((2,2),name="pool_4"))



model.add(Flatten())



model.add(Dense(activation="relu",units=128,

                kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.7,name="Drop_2"))

model.add(Dense(activation="sigmoid",units=1))



model.compile(optimizer=adama,loss = "binary_crossentropy", metrics=["accuracy"])
model.summary()
filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

xray_model = model.fit_generator(train_set,

                                steps_per_epoch = 163,

                                callbacks=callbacks_list,

                                epochs = 20,

                                validation_data = test_set,

                                validation_steps = 624//32)
model.save('model.h5')
test_accu = model.evaluate_generator(test_set,steps=624//32)
test_accu
# Accuracy 

plt.plot(xray_model.history['acc'])

plt.plot(xray_model.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='upper left')

plt.show()