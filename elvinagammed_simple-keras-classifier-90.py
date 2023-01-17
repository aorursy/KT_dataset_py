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
import numpy as np
import pandas as pd
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Activation
num_classes = 81
img_rows, img_cols = 32, 32
batch_size = 16
train_dir = "../input/fruits/fruits-360/Training/"
validation_dir = "../input/fruits/fruits-360/Test/"
# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  width_shift_range=0.3,
                                  height_shift_range=0.3,
                                  horizontal_flip=True,
                                  fill_mode="nearest")
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(img_rows, img_cols),
                                                   batch_size=batch_size,
                                                   class_mode="categorical",
                                                   shuffle=True)
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                   target_size=(img_rows, img_cols),
                                                   batch_size=batch_size,
                                                   class_mode="categorical",
                                                   shuffle=False)
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(img_rows, img_cols, 3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(131))
model.add(Activation("softmax"))
model.summary()
from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

checkpoint = ModelCheckpoint("fruit.h5", 
                            monitor="val_loss",
                            mode="min",
                            save_best_only=True,
                            verbose=1)
early_stop = EarlyStopping(monitor="val_loss", 
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.2,
                             patience=3,
                             verbose=1,
                             min_delta=0.0001)
callbacks = [early_stop, checkpoint]

model.compile(optimizer=RMSprop(lr=0.001), 
             loss="categorical_crossentropy",
             metrics=["accuracy"])
train_samples = 67692
test_samples = 22688
epochs = 5
model.fit_generator(train_generator, 
                    steps_per_epoch=train_samples // batch_size,
                       epochs=epochs,
                       callbacks=callbacks,
                       validation_data=validation_generator)
loss = pd.DataFrame({"loss": model.history.history['accuracy'],
                        "val_loss": model.history.history['val_accuracy']})
import matplotlib.pyplot as plt
plt.plot(loss['loss'])
plt.plot(loss['val_loss'])
