import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

from sklearn.metrics import f1_score, make_scorer

import numpy as np
import pandas as pd

import os
batch_size = 16
NAMA_PRAKTIKUM = "if4074-praktikum-1-cnn"
IMG_BASE_DIR = f'../input/{NAMA_PRAKTIKUM}'
IMG_TRAIN_DIR = f"{IMG_BASE_DIR}/P1_dataset/train"
IMG_TEST_DIR = f"{IMG_BASE_DIR}/P1_dataset/"

IMG_WIDTH = 256
IMG_HEIGHT = 256

CLASS_MODE = "categorical"

# this is the augmentation configuration we will use for training
# initial load, no preprocessing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.15
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        IMG_TRAIN_DIR,  # this is the target directory
        target_size=(IMG_WIDTH, IMG_HEIGHT),  # all images will be resized to 150x150
        batch_size=batch_size,
        subset="training",
        class_mode=CLASS_MODE  
) 

# this is a similar generator, for validation data
validation_generator = train_datagen.flow_from_directory(
        IMG_TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT), 
        batch_size=batch_size,
        subset="validation",
        class_mode=CLASS_MODE
)

# this is a similar generator, for test data
test_generator = test_datagen.flow_from_directory(
        IMG_TEST_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT), 
        # only read images from `test` directory
        classes=['test'],
        # don't generate labels
        class_mode=None,
        # don't shuffle
        shuffle=False,
)
if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
N_CLASS=4
    
# CONVOLUTIONAL LAYER
## CONV 1
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))

## CONV 2
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

# DENSE LAYER
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))

# OUTPUT LAYER
model.add(Dense(N_CLASS))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[
                  'accuracy',
                  tf.keras.metrics.Precision(),
                  tf.keras.metrics.Recall()
              ],)

model.summary()
N_CLASS=4
    
# CONVOLUTIONAL LAYER
## CONV 1
model_1 = Sequential()
model_1.add(Conv2D(32, (3, 3), input_shape=input_shape))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2))) # added

## CONV 2
model_1.add(Conv2D(64, (3, 3)))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) 

## CONV 3
model_1.add(Conv2D(128, (3, 3))) # added
model_1.add(Activation('relu')) # added
model_1.add(MaxPooling2D(pool_size=(2, 2))) # added

## CONV 3
model_1.add(Conv2D(255, (3, 3))) # added
model_1.add(Activation('relu')) # added
model_1.add(MaxPooling2D(pool_size=(2, 2))) # added

# DENSE LAYER
model_1.add(Flatten())
model_1.add(Dense(256))
model_1.add(Activation('relu'))
model_1.add(Dropout(0.5)) # added
model_1.add(Dense(256))
model_1.add(Activation('relu'))

# OUTPUT LAYER
model_1.add(Dropout(0.5)) # added
model_1.add(Dense(N_CLASS))
model_1.add(Activation('softmax'))
model_1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[
                  'accuracy',
                  tf.keras.metrics.Precision(),
                  tf.keras.metrics.Recall()
              ],)

model_1.summary()
CHOSEN_MODEL = "ripVGG-modified"

if CHOSEN_MODEL == "ripVGG-modified":    
    model = model_1
if CHOSEN_MODEL != "ripVGG-modified" and CHOSEN_MODEL != "ripVGG":
    raise Exception("CHOSEN_MODEL must be either ripVGG or ripVGG-modified")
epochs=80

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    )
import matplotlib.pyplot as plt

pd.Series(history.history['val_accuracy']).plot()
plt.title(f"Validation Accuracy over {epochs} epochs")
pd.Series(history.history['val_loss']).plot()
plt.title(f"Validation Loss over {epochs} epochs")
preds = model.predict_classes(
    test_generator)

def truncate_filename(x):
    return x.replace("test/", "")

submission = pd.DataFrame()
submission['id'] = pd.Series(test_generator.filenames).apply(truncate_filename)
submission['label'] = preds # np.argmax(preds, axis=1)
submission
try:
    SUB_NUM += 1
except:
    SUB_NUM = 0
    
submission.to_csv(f"submission{SUB_NUM}.csv", index=False)