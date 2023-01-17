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
import pandas as pd

import numpy as np

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import tensorflow_hub as hub

import tensorflow_datasets as tfds

from tensorflow.keras import layers

from PIL import Image

import tensorflow as tf

pd.options.display.max_rows = 999

pd.options.display.max_columns= 999

from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau

from matplotlib import pyplot
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
x_train = df_train.drop(['label'] , axis = 1)

y_train = df_train['label']

x_train.shape
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2 , random_state=42)
X_train.dtypes

X_train = X_train.values.reshape(X_train.shape[0], 28, 28 , 1).astype('float32')

X_test = X_test.values.reshape(X_test.shape[0], 28, 28 , 1).astype('float32')
X_train[1]

import matplotlib.pyplot as plt

print("the number of training examples = %i" % X_train.shape[0])

print("the number of classes = %i" % len(np.unique(y_train)))

print("Dimention of images = {:d} x {:d}  ".format(X_train[1].shape[0],X_train[1].shape[1])  )
sns.countplot(Y_test)

from keras.layers import Dropout

from keras import Sequential

from keras.layers import Dense, Dropout,Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

model = Sequential()



model.add(Conv2D(100, kernel_size=3, padding="valid", input_shape=(28, 28, 1), activation = 'relu'))

model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))

model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))



model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

from keras.layers.core import Activation



model.add(Flatten())

model.add(Dense(units= 500, activation='relu'  ))

model.add(Dropout(0.5))



model.add(Dense(10))

model.add(Activation("softmax"))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.utils import np_utils

Y_train = np_utils.to_categorical(Y_train).astype('int32')

Y_test = np_utils.to_categorical(Y_test)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])





from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=10,

    fill_mode='nearest',

    validation_split = 0.2

    )



datagen.fit(X_train)



train_generator = datagen.flow(X_train, Y_train, batch_size=60, subset='training')



validation_generator = datagen.flow(X_train, Y_train, batch_size=60, subset='validation')





# fits the model on batches with real-time data augmentation:

history = model.fit_generator(generator=train_generator,

                    validation_data=validation_generator,

                    use_multiprocessing=True,

                    steps_per_epoch = len(train_generator) / 60,

                    validation_steps = len(validation_generator) / 60,

                    epochs = 30,

                    workers=-1)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([-1,1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([-1,1.0])

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
test_set = (df_test.values).reshape(-1, 28, 28 , 1).astype('float32')

import numpy

res = model.predict(test_set)

res = numpy.argmax(res,axis = 1)

res = pd.Series(res, name="Label")

submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)

submission.to_csv("My_submission.csv",index=False)
