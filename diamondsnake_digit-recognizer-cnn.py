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
import tensorflow as tf

#from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten#, Dropout, Activation

from keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
train_df = pd.read_csv("../input/digit-recognizer/train.csv")

test_df = pd.read_csv("../input/digit-recognizer/test.csv")
height = 28

width = 28

classes = 10
# training images

X_train = train_df.drop('label', 1).to_numpy().reshape(len(train_df),height,width,1).astype('float32')

X_train /= 255



# training lables          

y_train = to_categorical(train_df['label'],classes)



# test images

X_test = test_df.to_numpy().reshape(len(test_df),height,width,1).astype('float32')

X_test /= 255
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
# visualise some training labels

f, axarr = plt.subplots(1,10, figsize=(20,20))

for i in range(10):

    image = X_train[i].reshape(28,28)

    axarr[i].axis('off')

    axarr[i].set_title("label = " + str(train_df['label'][i]))

    axarr[i].imshow(image, cmap='gray')
# compile model

model = Sequential()

model.add(Conv2D(32, kernel_size=(2,2), input_shape=(width, height,1), padding='same', activation='relu'))

model.add(MaxPooling2D())          

model.add(Conv2D(64, kernel_size=(2,2), padding='same', activation='relu'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dense(classes, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=20, validation_split=0.3)
accuracy = history.history['acc']

val_accuracy = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
score = model.evaluate(X_train, y_train)
predicted_classes = model.predict(X_test)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
# visualise some predicted labels

f, axarr = plt.subplots(1,10, figsize=(20,20))

for i in range(10):

    image = X_test[i].reshape(28,28)

    axarr[i].axis('off')

    axarr[i].set_title("prediction = " + str(predicted_classes[i]))

    axarr[i].imshow(image, cmap='gray')
ImageId = [x+1 for x in list(test_df.index)]

my_submission = pd.DataFrame({'ImageId': ImageId, 'Label': predicted_classes})

my_submission.to_csv('submission.csv', index=False)