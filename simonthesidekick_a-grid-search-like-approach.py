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

import matplotlib.pyplot as plt



from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import to_categorical



from sklearn.model_selection import train_test_split
sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

train = pd.read_csv("../input/Kannada-MNIST/train.csv")



x = train.iloc[:,1:].values

y = train.iloc[:,0].values

x_test = test.drop('id', axis=1).iloc[:,:].values

x = x.reshape(x.shape[0], 28, 28, 1)

y = to_categorical(y, 10)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)



x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.10, random_state=42) 



train_datagen = ImageDataGenerator(rescale=1./255.,

                                   rotation_range=10,

                                   width_shift_range=0.10,

                                   height_shift_range=0.10,

                                   shear_range=0.1,

                                   zoom_range=0.1,

                                   horizontal_flip=False)



valid_datagen = ImageDataGenerator(rescale=1./255.)
optimizer = RMSprop(lr=0.001)



model= Sequential()

model.add(Conv2D(24,kernel_size = 3, padding = "same", activation = "relu",input_shape = (28,28,1)))

model.add(Conv2D(24,kernel_size = 3, padding = "same", activation = "relu"))

model.add(Conv2D(48 ,kernel_size = 5, padding = "same", activation = "relu"))          

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))

model.add(Dropout(0.1))

model.add(Conv2D(48,kernel_size = 3, padding = "same", activation = "relu"))

model.add(Conv2D(48,kernel_size = 3, padding = "same", activation = "relu"))

model.add(Conv2D(96 ,kernel_size = 5, padding = "same", activation = "relu"))          

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))

model.add(Dropout(0.1))

model.add(Conv2D(96,kernel_size = 3, padding = "same", activation = "relu"))

model.add(Conv2D(96,kernel_size = 3, padding = "same", activation = "relu"))

model.add(Conv2D(192 ,kernel_size = 5, padding = "same", activation = "relu"))          

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))

model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(256))

model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics = ["accuracy"])


learning_rate_reduction= ReduceLROnPlateau(monitor='val_loss', 

                                            patience=1,

                                            verbose=1,

                                            factor=0.5,

                                            min_lr=0.0001)





early = EarlyStopping(monitor='val_loss', min_delta=0.0005, mode='min', verbose=1, patience=2)

history = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=1024),

                                  steps_per_epoch=100,

                                  epochs=30,

                                  validation_data=valid_datagen.flow(x_valid, y_valid),

                                  validation_steps=50,

                                  callbacks=[learning_rate_reduction])
predictions = model.predict_classes(x_test/255.)

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.to_csv("submission.csv", index=False)