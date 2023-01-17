# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras import models

from keras import layers

from keras import optimizers

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt
TRAIN_PATH = "../input/pokemon_train.npy"

TEST_PATH = "../input/pokemon_test.npy"

def preprocess_data():

    train_data = np.load(TRAIN_PATH)

    



    # random

    rng = np.random.RandomState(SEED)

    indices = np.arange(len(train_data))

    rng.shuffle(indices)

    train_data = train_data[indices]



    val_data = train_data[:100]



    train_data = train_data[100:]



    x_train = train_data[:,1:]

    y_train = train_data[:,0]



    x_val = val_data[:,1:]

    y_val = val_data[:,0]



    return x_train, y_train, x_val, y_val
def build_model():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),

                    loss='categorical_crossentropy',

                    metrics=['accuracy'])



    print(model.summary())

    return model
SEED = 113

model = build_model()

x_train, y_train, x_val, y_val = preprocess_data()

y_train = to_categorical(y_train)

y_val = to_categorical(y_val)



train_datagen = ImageDataGenerator(rescale=1./255)

validation_datagen = ImageDataGenerator(rescale=1./255)



x_train = np.reshape(x_train, (x_train.shape[0], 128,128,3))

x_val = np.reshape(x_val, (x_val.shape[0], 128,128,3))



train_datagen.fit(x_train)

validation_datagen.fit(x_val)



model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=20),

                  steps_per_epoch=100, 

                  epochs=30,

                  validation_data=validation_datagen.flow(x_val, y_val, batch_size=20),

                  validation_steps=20)

    

test_data = np.load(TEST_PATH)

x_test = np.reshape(test_data, (test_data.shape[0], 128,128,3))

x_test = x_test / 255.

predict_labels = model.predict_classes(x_test, batch_size=32)

print(predict_labels)

predict_label_csv = np.hstack([(np.arange(predict_labels.shape[0])+1).reshape([-1, 1]), predict_labels.reshape([-1, 1])])

np.savetxt('predict_label.csv', predict_label_csv, delimiter = ',', header='Id,Category')