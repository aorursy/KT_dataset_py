import keras

from keras.datasets import mnist

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd
batch_size = 128

num_classes = 10

epochs = 30
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
y_train = train["label"]



x_train = train.drop(labels=["label"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train , random_state = 42, test_size=0.3)
x_train.shape
x_test.shape
y_train.shape, y_test.shape
from keras import backend
if backend.image_data_format() == 'channels_first':

    x_train = np.array(x_train).reshape(x_train.shape[0], 1, 28, 28)

    x_test = np.array(x_test).reshape(x_test.shape[0], 1, 28, 28)

    input_shape = (1, 28, 28)

else:

    x_train = np.array(x_train).reshape(x_train.shape[0], 28, 28, 1)

    x_test = np.array(x_test).reshape(x_test.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)
x_train.shape
test = np.array(test).reshape(-1, 28, 28, 1)
x_train = x_train.astype('float')

x_test = x_test.astype('float')

test = test.astype('float')



x_train /= 255

x_test /= 255

test/= 255
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
from keras.models import Sequential

model = Sequential()
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(lr = 0.001, rho=0.9, epsilon=1e-08, decay = 0.0), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.3, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
from keras.callbacks import ReduceLROnPlateau

annealer_callback = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)
model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size = batch_size), steps_per_epoch= x_train.shape[0] // batch_size, epochs=epochs, validation_data=(x_test,y_test), verbose = 2,

                    callbacks = [annealer_callback])
score = model.evaluate(x_test, y_test, verbose=0)

score
results = model.predict(test)



results = np.argmax(results,axis = 1)
submission_df = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
submission_df["Label"] = results
submission_df.to_csv("submission.csv", index = False)