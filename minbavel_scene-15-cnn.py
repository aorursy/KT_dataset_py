import keras

from keras.models import Sequential

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras import regularizers, optimizers

from keras.layers.core import Lambda

import numpy as np

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import numpy as np

import cv2

import os

from tqdm import tqdm_notebook as tqdm



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.regularizers import l2

!pip install imutils

from imutils import paths

import argparse
imagePaths = list(paths.list_images("/kaggle/input/15-scene/15-Scene"))

data = []

labels = []



for imagePath in imagePaths:

    label = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath, 2)

    image = cv2.resize(image, (224, 224 ))

    data.append(np.reshape(image, [224,224,1]))

    labels.append(label)

    

data = np.array(data, dtype="float") / 255.0



lb = LabelBinarizer()

labels = lb.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.10, stratify=labels, random_state=42)



print(trainX.shape)

print(trainY.shape)

print(testX.shape)

print(testY.shape)
baseMapNum = 32

model = Sequential()

model.add(Conv2D(baseMapNum, (3,3), padding='same', input_shape=trainX.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(baseMapNum, (3,3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(2*baseMapNum, (3,3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(2*baseMapNum, (3,3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))



model.add(Conv2D(4*baseMapNum, (3,3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(4*baseMapNum, (3,3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(15, activation='softmax'))



model.summary()
datagen = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center=False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    zca_whitening=False,

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    vertical_flip=False

    )

datagen.fit(trainX)
batch_size = 64

epochs=200



opt_rms = keras.optimizers.rmsprop(lr=0.0001,decay=1e-5)

model.compile(loss='categorical_crossentropy',

        optimizer=opt_rms,

        metrics=['accuracy'])

hist = model.fit_generator(datagen.flow(trainX, trainY, batch_size=batch_size),

                           steps_per_epoch=trainX.shape[0] // batch_size,epochs=epochs,verbose=2,

                           validation_data=(testX,testY))



scores = model.evaluate(testX, testY, verbose=2)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])

print("\nTraining Accuracy: %.2f%%" % (hist.history['acc'][epochs - 1]*100))

print('Test result: %.3f \nloss: %.3f' % (scores[1]*100,scores[0]))
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.legend(['Train','Test'])

plt.title('Loss')

plt.savefig("loss.png",dpi=300,format="png")

plt.figure()

plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.legend(['Train','Test'])

plt.title('Accuracy')

plt.savefig("accuracy.png",dpi=300,format="png")