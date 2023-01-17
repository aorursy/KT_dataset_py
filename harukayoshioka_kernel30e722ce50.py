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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
x_train = train.drop('label', 1)

y_train = train['label']
y_train.value_counts()
x_train.shape, y_train.shape
x_train = x_train.values.reshape(-1, 28, 28, 1)
y_train = y_train.values.reshape(-1, 1)
y_train.shape
x_test = test.values.reshape(-1, 28, 28, 1)
x_test.shape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(

        rescale=1.0/255,

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
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
x_train.shape, x_val.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

from tensorflow.keras.callbacks import ModelCheckpoint
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(rate=0.25))

model.add(Dense(10, activation='softmax'))
model.summary()
batch_size = 128

epochs = 20
model_path = 'model/tmp-best-model.h5'
checkpoint = ModelCheckpoint(model_path, 

                             verbose=1, 

                             monitor='val_loss',

                             save_best_only=True,

                             save_weight_only=True, 

                             mode='auto')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val, y_val),

                              steps_per_epoch=x_train.shape[0] // batch_size,

                              callbacks=[checkpoint],

                             )



















































plt.figure(figsize=(12, 6))

plt.plot(np.arange(20)+1, history.history['loss'], label='Loss')

plt.plot(np.arange(20)+1, history.history['val_loss'], label='Validation Loss')

plt.title('losses over training', fontsize=20)



plt.xlabel('epochs', fontsize=15)

plt.ylabel('loss', fontsize=15)



plt.legend()

plt.show()
plt.figure(figsize=(12, 6))

plt.plot(np.arange(20)+1, history.history['accuracy'], label='Accuracy')

plt.plot(np.arange(20)+1, history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Accuracy over training', fontsize=20)



plt.xlabel('epochs', fontsize=15)

plt.ylabel('Accuracy', fontsize=15)



plt.legend()

plt.show()
result = model.predict(x_test)
result = np.argmax(result, axis=1)
result = pd.Series(result, name='label')
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission['Label'] = result
submission['Label'].value_counts()
sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')