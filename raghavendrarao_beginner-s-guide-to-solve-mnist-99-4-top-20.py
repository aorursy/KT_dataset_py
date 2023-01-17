# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the Train and Test Datasets.

mnist_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

mnist_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(mnist_train.shape, mnist_test.shape)
# Looking at a few rows from the data isn't a bad idea.

mnist_train.head()
img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = keras.utils.np_utils.to_categorical(raw.label, num_classes)



    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y



x, y = data_prep(mnist_train)



X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.10, random_state=42)


ax = plt.subplots(1,5)

for i in range(0,5):   #validate the first 5 records

    ax[1][i].imshow(X_train[i][:,:,0])

    
datagen = ImageDataGenerator(

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

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3, 3),padding = 'Same',

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(BatchNormalization(axis=1))

model.add(Conv2D(filters = 32, kernel_size=(3, 3),padding = 'Same', activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.4))

model.add(BatchNormalization(axis=1))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(axis=1))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(BatchNormalization(axis=1))

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=128),

          epochs=22,callbacks=[learning_rate_reduction],

          validation_data = (X_val,y_val))
from matplotlib import pyplot

pyplot.subplot(211)

pyplot.title('Categorial Cross Entropy Error')

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()

pyplot.subplot(211)

pyplot.title('Accuracy')

pyplot.plot(history.history['accuracy'], label='train')

pyplot.plot(history.history['val_accuracy'], label='test')

pyplot.legend()

pyplot.show()
mnist_test.head()


x_test_shaped_array = mnist_test.values.reshape(28000, img_rows, img_cols, 1)

x_test = x_test_shaped_array / 255
# predict results

results = model.predict(x_test)

# select the indix with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist.csv",index=False)