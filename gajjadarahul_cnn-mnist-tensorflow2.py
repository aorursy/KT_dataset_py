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
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
train.shape, test.shape
y_train = train['label']
X_train = train.drop(['label'], axis=1)
X_test = test
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
X_train = X_train/255.
X_test = X_test/255.
X_train.shape, y_train.shape, X_test.shape
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D, BatchNormalization, LeakyReLU, Concatenate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import LeakyReLU
y_train = to_categorical(y_train)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state=1250)
# CNN Model
model = Sequential(name = 'cnn_mnist')

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
# model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation ='relu'))
# model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'Same', activation ='relu'))
# model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation ='relu'))
# model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
opt = optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
datagen = ImageDataGenerator(
                            rotation_range=15,
                            width_shift_range=0.25,
                            height_shift_range=0.25,
                            zoom_range=0.25,
                            horizontal_flip=False,
                            vertical_flip=False
                            )
datagen.fit(X_train)
rls = ReduceLROnPlateau(monitor='accuracy', mode = 'max', factor=0.5, min_lr=1e-7, verbose = 1, patience=5)
es = EarlyStopping(monitor='accuracy', mode='max', verbose = 1, patience=50)
mc = ModelCheckpoint('cnn_best_model.h5', monitor='accuracy', mode='max', verbose = 1, save_best_only=True)

callback_list = [rls, es, mc]
batchSize = 128
n_epochs = 200
model.fit_generator(datagen.flow(X_train, y_train, batch_size = batchSize),
                                 validation_data = (X_valid, y_valid),
                                 steps_per_epoch = X_train.shape[0] // batchSize, 
                                 epochs = n_epochs, 
                                 verbose = 1,
                                 callbacks = callback_list)
def plot_model(history): 
    fig, axs = plt.subplots(1,2,figsize=(16,8)) 
    # summarize history for accuracy
    axs[0].plot(history.history['accuracy'], 'c') 
    axs[0].plot(history.history['val_accuracy'],'m') 
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'validate'], loc='upper left')
    # summarize history for loss
    axs[1].plot(history.history['loss'], 'c') 
    axs[1].plot(history.history['val_loss'], 'm') 
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss') 
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validate'], loc='upper right')
    plt.show()
# Version 3 -> 200 epochs 128 batch size
plot_model(model.history)
# Version 3 -> 100 epochs
plot_model(model.history)
# Version 2 -> 50 epochs 0.25 data augmentation
plot_model(model.history)
# Version 1 -> 0.125 augmentation
plot_model(model.history)
best_model = load_model('cnn_best_model.h5')

y_pred = best_model.predict_classes(X_test, verbose=0)

final = pd.DataFrame({"ImageId": list(range(1, len(y_pred)+1)),
                          "Label": y_pred})

final.to_csv("submission_digit_recognizer.csv", index=False)

