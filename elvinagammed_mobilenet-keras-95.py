# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.applications import MobileNet
img_rows, img_cols = (224, 224)
num_classes = 10
MobileNet = MobileNet(weights='imagenet', 
                     include_top=False,
                     input_shape=(img_rows, img_cols, 3))
for layer in MobileNet.layers:
    layer.trainable=False
for (i, layer) in enumerate(MobileNet.layers):
    print(str(i) + layer.__class__.__name__)
#     layer.trainable
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
transfer_model = Sequential()
for layer in MobileNet.layers:
    transfer_model.add(layer)
transfer_model.add(GlobalAveragePooling2D())
transfer_model.add(Dense(1024, activation="relu"))  
transfer_model.add(Dense(1024, activation="relu"))  
transfer_model.add(Dense(512, activation="relu"))  
transfer_model.add(Dense(10, activation="softmax")) 
transfer_model.summary()
train_dir = "../input/10-monkey-species/training/training/"
validation_dir = "../input/10-monkey-species/validation/validation/"
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=50, width_shift_range=0.5, height_shift_range=0.5, 
                                  horizontal_flip=True, fill_mode="nearest")
validation_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 16
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_rows, img_cols), batch_size=batch_size, 
                                                    class_mode="categorical")
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(img_rows, img_cols), batch_size=batch_size, 
                                                    class_mode="categorical")
# Train
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=3, min_delta=0, verbose=1, restore_best_weights=True)
transfer_model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["accuracy"])
epochs = 35
batch_size = 16
history = transfer_model.fit_generator(train_generator, steps_per_epoch=1098//batch_size, epochs = epochs, callbacks=early_stop,
                                       validation_data=validation_generator, validation_steps=272//batch_size)
import pandas as pd
loss = pd.DataFrame(transfer_model.history.history)

loss['accuracy'].plot()
loss['val_accuracy'].plot()
