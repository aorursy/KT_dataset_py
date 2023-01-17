# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras import optimizers

import numpy as np

from keras.layers.core import Lambda

from keras import backend as K

from keras import regularizers

# import pandas as pd

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
train_data = pd.read_csv("../input/train.csv")
num_classes = 10
def process_data(data):

    if "label" in data.columns:

        data_x = data.drop("label", axis=1)

        data_y = data["label"]

        data_x = data_x.values.reshape(data.shape[0], 28, 28, 1)

        return (data_x, data_y)

    else:

        data_x = data.values.reshape(data.shape[0], 28, 28, 1)

        return data_x
train_x, train_y = process_data(train_data)

print(train_x.shape, train_y.shape)
train_x = train_x.astype("float32") / 255.0
train_x_mean = np.mean(train_x, axis=0)

train_x -= train_x_mean
train_y = keras.utils.to_categorical(train_y, num_classes)
seed = 2

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1, random_state=seed)
print(type(train_x))
INPUT_SHAPE = train_x.shape[1:]



model = Sequential()



model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=INPUT_SHAPE))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))    

model.add(Flatten())



model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
MODEL_PATH="model.h5"

checkpoint = ModelCheckpoint(filepath=MODEL_PATH,

                            monitor='val_acc',

                            verbose=1, 

                            save_best_only=True)

callbacks = [checkpoint]
batch_size = 128

epochs = 20
datagen = ImageDataGenerator(rotation_range=10,

                            horizontal_flip=True,

                            validation_split=0.1)

datagen.fit(train_x)

train_generator = datagen.flow(train_x, train_y, batch_size=batch_size)
model.fit_generator(train_generator,

                   steps_per_epoch=train_x.shape[0] // batch_size,

                   validation_data=(test_x, test_y),

                   epochs=epochs,

                   verbose=1, 

                   workers=4, 

                   callbacks=callbacks)