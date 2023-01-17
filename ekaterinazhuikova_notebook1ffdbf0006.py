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

from tensorflow.keras.datasets import cifar10, fashion_mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import utils

from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt

%matplotlib inline
from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt

%matplotlib inline 

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']
X_train.shape
X_test.shape
batch_size = 256

nb_classes = 10

nb_epoch = 40

img_rows, img_cols = 32, 32

img_channels = 3
X_train = X_train.reshape((50000, 32, 32, 3))

X_train = X_train.astype('float32')

X_test = X_test.reshape((10000, 32, 32, 3))

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
Y_train = utils.to_categorical(y_train, nb_classes)

Y_test = utils.to_categorical(y_test, nb_classes)
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(BatchNormalization())



model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Flatten()) 





model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu'))

model.add(Dense(nb_classes, activation='softmax'))
callbacks_list = [EarlyStopping(monitor='val_loss', patience=5),

                  ModelCheckpoint(filepath='my_model.h5',

                                  monitor='val_loss',

                                  save_best_only=True),

                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

                  ] 
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit(X_train, Y_train,

              batch_size=batch_size,

              epochs=nb_epoch,

              callbacks=callbacks_list,

              validation_split=0.1,

              verbose=1)
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))
plt.plot(history.history['accuracy'], 

         label='Доля правильных ответов на обучающем наборе')

plt.plot(history.history['val_accuracy'], 

         label='Доля правильных ответов на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Доля правильных ответов')

plt.legend()

plt.show()
plt.plot(history.history['loss'], 

         label='Оценка потерь на обучающем наборе')

plt.plot(history.history['val_loss'], 

         label='Оценка потерь на проверочном наборе')

plt.xlabel('Эпоха обучения')

plt.ylabel('Оценка потерь')

plt.legend()

plt.show()