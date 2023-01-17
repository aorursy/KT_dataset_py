# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as p # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import models, Model, Input, layers, callbacks
def prepare_data():
    
    train_dir = '/kaggle/input/sign-language-mnist/sign_mnist_train.csv'
    test_dir = '/kaggle/input/sign-language-mnist/sign_mnist_test.csv'
    
    df_train = p.read_csv(train_dir)
    df_test = p.read_csv(test_dir)
    y_train = to_categorical(df_train['label'])
    y_test = to_categorical(df_test['label'])
    del df_train['label']
    del df_test['label']
    
    df_train = df_train.values / 255
    df_test = df_test.values / 255
    x_train = df_train.reshape(df_train.shape[0], 28, 28,1)
    x_test = df_test.reshape(df_test.shape[0], 28, 28, 1)
    
    
    
    return x_train, y_train, x_test, y_test
x_train, y_train, x_test, y_test = prepare_data()
#----COPIED----
data_gen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)
#--------------
data_gen.fit(x_train)

my_callbacks = [callbacks.ReduceLROnPlateau(factor= 0.5, metrics='val_accuracy', patience=3),
            callbacks.ModelCheckpoint(mointor='val_accuracy', filepath='/best0.hdf5', save_best_only=True)]
model0 = models.Sequential()

model0.add(layers.Conv2D(256, (3,3),activation='relu', input_shape=(28, 28, 1)))
model0.add(layers.Conv2D(256, (3,3), activation='relu'))
model0.add(layers.BatchNormalization())
model0.add(layers.MaxPooling2D())
model0.add(layers.Dropout(0.3))

model0.add(layers.Conv2D(128, (3,3), activation='relu'))
model0.add(layers.Conv2D(128, (3,3), activation='relu'))
model0.add(layers.BatchNormalization())
model0.add(layers.MaxPooling2D())
model0.add(layers.Dropout(0.3))


model0.add(layers.Flatten())

model0.add(layers.Dense(256, activation='relu'))
model0.add(layers.Dense(25, activation='softmax'))

model0.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model0.fit(data_gen.flow(x_train, y_train, batch_size=128), epochs=20, validation_data=(x_test, y_test), callbacks=my_callbacks)
model0.load_weights('/best0.hdf5')
model0.evaluate(x_test, y_test)
