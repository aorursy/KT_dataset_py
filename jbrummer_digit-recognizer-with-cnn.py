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
import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator



img_rows,img_cols = 28,28



train = pd.read_csv("../input/digit-recognizer/train.csv")

x_test = pd.read_csv("../input/digit-recognizer/test.csv")



y_train = train["label"]

x_train = train.drop(labels = ["label"],axis = 1)



x_test /= 255.0

x_train /= 255.0



x_train = x_train.values.reshape((-1,28,28,1))

x_test = x_test.values.reshape((-1,28,28,1))



y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2)



model = Sequential()



model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (img_rows, img_cols, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

    

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(128, kernel_size = 4, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["acc"])

from keras.callbacks import ReduceLROnPlateau



data_gen = ImageDataGenerator(

            rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1,

        horizontal_flip=False,

        fill_mode='nearest')

data_gen.fit(x_train)



learning_rate = tf.keras.callbacks.ReduceLROnPlateau(

monitor='val_acc',

patience=3,

factor=0.5,

min_lr=0.00001)



model.fit_generator(data_gen.flow(x_train,y_train, batch_size=86),

                              epochs = 50, validation_data=(x_val,y_val), 

                              steps_per_epoch=x_train.shape[0] // 86,

                              callbacks=[learning_rate])
results = model.predict(x_test)

results = np.argmax(results, axis=1)

results = pd.Series(results, name='Label')



submission = pd.concat([pd.Series(range(1,28001), name='ImageId'), results], axis=1)

submission.to_csv(r'Digit_Recognizer', index=False)