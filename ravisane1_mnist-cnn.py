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

import numpy as np

import pandas as pd

import os



from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from sklearn.model_selection import train_test_split



print(os.listdir("../input/digit-recognizer"))
# load the train and test csv files

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
# for labels 

train_y = train['label']

# for pixel values

train_x = train.drop(labels=['label'], axis=1)
# normalize the data

train_x = train_x / 255.0

test = test / 255.0
train_x = train_x.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)
train_X, val_X, train_Y, val_Y = train_test_split(train_x, train_y, test_size = 0.2)
input_shape = (28, 28, 1)
model = tf.keras.Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))

# model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.summary()
model.compile(

    optimizer = 'adam',#tf.keras.optimizers.RMSprop(),

    loss = tf.keras.backend.categorical_crossentropy, 

    metrics = ['accuracy']

)
# This callback will stop the training when there is no improvement in

# the validation accuracy for three consecutive epochs.

callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.00001)

history = model.fit(train_X, train_Y, 

          batch_size=32,

          validation_data=(val_X, val_Y),

          epochs=30, callbacks=[callback])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# predict results

results = model.predict(test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results], axis = 1)



submission.to_csv("submission.csv",index=False)