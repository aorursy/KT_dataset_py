# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pathlib import Path



%matplotlib inline

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras

import numpy as np

import pandas as pd



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import LearningRateScheduler
def build_model(optimizer=Adam()):

    model = Sequential()



    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128, kernel_size=3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(256, kernel_size=3, activation='relu'))

    model.add(BatchNormalization())

    model.add(GlobalMaxPooling2D())

    model.add(Dropout(0.2))



    model.add(Flatten())

    model.add(Dense(1024))

    model.add(BatchNormalization())

    model.add(Dense(1024))

    model.add(BatchNormalization())

    model.add(Dense(512))

    model.add(BatchNormalization())

    model.add(Dense(256))

    model.add(BatchNormalization())

    model.add(Dense(128))

    model.add(BatchNormalization())

    model.add(Dense(10, activation='softmax'))

    

    model.compile(loss='sparse_categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])

    

    return model
model = build_model(Adam(learning_rate=1e-3))
weightspath = Path.cwd()/'..'/'input'/'pretrained-larger-classifier-kannadamnist'/'model_fullTraining_initialPadding.h5'

model.load_weights(str(weightspath))
test_path = Path.cwd()/'..'/'input'/'Kannada-MNIST'/'test.csv'



all_data_test = pd.read_csv(test_path)
x_test = all_data_test.iloc[:,1:].values

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
predictions = model.predict_classes(x_test/255.)
predictions[:10]
output = pd.DataFrame({'id': all_data_test.id,

                       'label': predictions})

output.to_csv("submission.csv",index=False)