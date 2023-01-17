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
train_dir = '/kaggle/input/intel-image-classification/seg_train/seg_train/'
prediction_dir = '/kaggle/input/intel-image-classification/seg_pred/seg_pred/'
test_dir = '/kaggle/input/intel-image-classification/seg_test/seg_test/'
# plot one random image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

img1 = mpimg.imread(os.path.join(train_dir, 'street/5713.jpg'))
img2 = mpimg.imread(os.path.join(prediction_dir, '1614.jpg'))
img3 = mpimg.imread(os.path.join(test_dir, 'forest/20330.jpg'))

fig, axis = plt.subplots(1,3, figsize=(15,10))
axis[0].imshow(img1)
axis[0].set_xlabel("Train set image")
axis[1].imshow(img2)
axis[1].set_xlabel("Prediction set image")
axis[2].imshow(img3)
axis[2].set_xlabel("Test set image")
plt.show()
sizes = []
sizes.append(img1.shape)
sizes.append(img2.shape)
sizes.append(img3.shape)
print(sizes)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1. /255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(150,150),
                                                   batch_size=64)
test_datagen = ImageDataGenerator(rescale = 1. /255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                   target_size=(150,150),
                                                   batch_size=64)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
def VGG16():
    model = Sequential()
    model.add(Conv2D(input_shape=(150,150,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(Adam(lr=0.1), loss = "categorical_crossentropy", metrics = ['accuracy'])
    return model
model = Sequential([
    Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 
    MaxPool2D(2,2),
    Conv2D(32, (3, 3), activation = 'relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')
])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor="val_loss",
                               min_delta=0,
                               patience=2,
                               verbose=0,
                               mode="auto",
                               baseline=None,
                               restore_best_weights=False)
steps_per_epoch = train_generator.n / train_generator.batch_size
validation_steps = test_generator.n / test_generator.batch_size
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data = test_generator,
                    validation_steps = validation_steps,
                    epochs=15,
                    verbose=1,
                    callbacks = [early_stopping])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()
model = VGG16()
print(model.summary())
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor="val_loss",
                               min_delta=0,
                               patience=0,
                               verbose=0,
                               mode="auto",
                               baseline=None,
                               restore_best_weights=False,)

steps_per_epoch = train_generator.n / train_generator.batch_size
validation_steps = test_generator.n / test_generator.batch_size
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data = test_generator,
                    validation_steps = validation_steps,
                    epochs=15,
                    verbose=1,
                    callbacks = [early_stopping])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()