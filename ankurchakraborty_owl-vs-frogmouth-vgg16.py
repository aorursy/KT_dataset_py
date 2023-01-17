import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import glob
import shutil
# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

#create new folder structure usable by keras image processor
DATA_DIR = "data"
TRAIN_DIR = "train"
TEST_DIR = "test"
OWL = "OWL"
FROGMOUTH = "FROGMOUTH"

GENERATED_DIR = "GEN"

os.makedirs(os.path.join(os.getcwd(),DATA_DIR, TRAIN_DIR,OWL), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(),DATA_DIR, TRAIN_DIR,FROGMOUTH), exist_ok=True)

os.makedirs(os.path.join(os.getcwd(),DATA_DIR, TEST_DIR,OWL), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(),DATA_DIR, TEST_DIR,FROGMOUTH), exist_ok=True)


os.makedirs(os.path.join(os.getcwd(),DATA_DIR, GENERATED_DIR), exist_ok=True)


#copy existing data images into new directory
#copy training frogmouth
files = glob.iglob('../input/owl-and-frogmouth/owl_and_frogmouth/train/frogmouth*.*')

for path in files:
    if(path.endswith('csv')):
        continue
    shutil.copy(path,os.path.join(os.getcwd(),DATA_DIR, TRAIN_DIR,FROGMOUTH))
 
#copy test frogmouth
files = glob.iglob('../input/owl-and-frogmouth/owl_and_frogmouth/test/frogmouth*.*')
for path in files:
    if(path.endswith('csv')):
        continue
    shutil.copy(path,os.path.join(os.getcwd(),DATA_DIR, TEST_DIR,FROGMOUTH))
    
#copy training owl
files = glob.iglob('../input/owl-and-frogmouth/owl_and_frogmouth/train/owl*.*')
for path in files:
    if(path.endswith('csv')):
        continue
    
    shutil.copy(path,os.path.join(os.getcwd(),DATA_DIR, TRAIN_DIR,OWL))
    
#copy test owl
files = glob.iglob('../input/owl-and-frogmouth/owl_and_frogmouth/test/owl*.*')
for path in files:
    if(path.endswith('csv')):
        continue
    
    shutil.copy(path,os.path.join(os.getcwd(),DATA_DIR, TEST_DIR,OWL))

plt.show()

#our set image size
IMAGE_SHAPE = [150,150]


#instantiate pre trained VGG16 model with imagenet weights prefilled
pre_t_model = VGG16(
                input_shape= IMAGE_SHAPE + [3],
                weights='imagenet',
                include_top = False)

#we don't want to retrain the conv layers
pre_t_model.trainable = False

K = 2 #two classes
x = Flatten() (pre_t_model.output)
x = Dense(2048, activation='relu') (x)
x = Dropout(0.2) (x)
x = Dense(K, activation='softmax') (x)

#create our model
model = Model(inputs=pre_t_model.input, outputs=x)

model.summary()
#creating image data generators
gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        preprocessing_function=preprocess_input)

batch_size = 8

#training generator
train_generator = gen.flow_from_directory(
                os.path.join(os.getcwd(),DATA_DIR, TRAIN_DIR),
                shuffle=True,
                target_size=IMAGE_SHAPE,
                batch_size=batch_size,
                save_to_dir=os.path.join(os.getcwd(),DATA_DIR,GENERATED_DIR),
                save_prefix='train_gen')

val_generator = gen.flow_from_directory(
                os.path.join(os.getcwd(),DATA_DIR, TEST_DIR),
                shuffle=True,
                target_size=IMAGE_SHAPE,
                batch_size=batch_size,
                save_to_dir=os.path.join(os.getcwd(),DATA_DIR,GENERATED_DIR),
                save_prefix='test_gen')


#compile and train model

opti = SGD(lr=0.0001, momentum=0.8)
#opti = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(
        loss='categorical_crossentropy',
        optimizer=opti,
        metrics=['accuracy'])

r = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=3,
        epochs=30,
        validation_steps=3)
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')

plt.legend()
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
pre_t_model.trainable = True

set_trainable = False
for layer in pre_t_model.layers:
    if layer.name == "block5_conv1":
        set_trainable = True
    layer.trainable = set_trainable
    
model.summary()


#compile and train model

opti = SGD(lr=0.000001)
#opti = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model.compile(
        loss='categorical_crossentropy',
        optimizer=opti,
        metrics=['accuracy'])


#now train the model
r = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=3,
        epochs=100,
        validation_steps=3)
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val accuracy')

plt.legend()
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()