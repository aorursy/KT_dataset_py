# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import BatchNormalization



from keras.callbacks import LearningRateScheduler

from keras.preprocessing.image import ImageDataGenerator



import matplotlib.pyplot as plt

from PIL import Image

import skimage.morphology as skm



import sklearn.utils

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import glob

import gc

print(os.listdir("../input/cell_images/cell_images"))



# Any results you write to the current directory are saved as output.
# Parameter section

TARGET_IMAGE_SIZE = 50 

VALIDATION_SIZE_PERCENTAGE = 0.2

BATCH_SIZE = 64

NUM_EPOCH = 25



USE_DATA_AUGMENTATION = False

EMPHASIZE_DARK_SPOTS = False

def read_images(folder, label, image_data, label_data):

    for fname in glob.glob("../input/cell_images/cell_images/" + folder + "/*.png"):

        img = Image.open(fname).resize((TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))

        img = np.array(img).astype("float32") / 255

        if EMPHASIZE_DARK_SPOTS:

            img = skm.black_tophat(img)

        image_data.append(img)

        label_data.append(label)



image_data = []

label_data = []

read_images("Parasitized", 1, image_data, label_data)

read_images("Uninfected", 0, image_data, label_data)

image_data = np.array(image_data)
# show an image to control the importing

n=1284

plt.imshow(image_data[n])

print(label_data[n])
# split randomly into a training and validation set

X_train, X_val, Y_train, Y_val = train_test_split(

    image_data, label_data, test_size=VALIDATION_SIZE_PERCENTAGE, shuffle=True)
def build_model():

    model = Sequential()

    model.add(Conv2D(32, 

                 kernel_size=(3,3), 

                 activation = "relu", 

                 input_shape = (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3)

    ))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(32,(3,3),activation="relu"))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(64,(3,3),activation="relu"))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(128,activation = "relu"))

    model.add(Dropout(0.25))



    model.add(Dense(1, activation = "sigmoid"))

    return model
model = build_model()

model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])

model.summary()
lrs = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)



if USE_DATA_AUGMENTATION:

    datagen = ImageDataGenerator(

        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),

                              epochs = NUM_EPOCH, 

                              validation_data = (X_val, Y_val),

                              verbose = 1, 

                              callbacks=[lrs],

                              steps_per_epoch=len(X_train) // BATCH_SIZE)

else:

    history = model.fit(X_train, Y_train,

                    validation_data=(X_val, Y_val),

                    batch_size=BATCH_SIZE,

                    epochs=NUM_EPOCH,

                    callbacks=[lrs],

                    verbose=1)
def show_accuracy_vs_epoch(history):

    xlabel = 'Epoch'

    legends = ['Training', 'Validation']

    plt.xlabel(xlabel, fontsize=13)

    plt.ylabel('Accuracy', fontsize=13)

    if USE_DATA_AUGMENTATION:

        title = "Training with data augmentation (flip, rotate)"

    else:

        title = "Training without data augmentation"

    plt.title(title)

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.legend(legends, loc='lower right')



show_accuracy_vs_epoch(history)
# Calculate errors and confusion matrix for the validation data

Y_pred_probabilities = model.predict(X_val)

Y_pred = np.round(Y_pred_probabilities)



errors = []

for i in range(len(Y_pred)):

    if Y_pred[i] != Y_val[i]:

        errors.append([i, Y_pred_probabilities[i][0]])

sorted_errors = sorted(errors, key=lambda rec: rec[1], reverse=True)



print(confusion_matrix(Y_val, Y_pred))

print("\nNumber of errors = ", len(sorted_errors))

print("Accuracy = ", (len(Y_val) - len(sorted_errors)) / float(len(Y_val)))
def display_errors(errors_index):

    """Show images with their predicted and real labels."""

    n = 0

    nrows = 4

    ncols = 5

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(10,10))

    for row in range(nrows):

        for col in range(ncols):

            index = errors_index[n][0]

            pred = errors_index[n][1]

            img = X_val[index]

            ax[row, col].imshow(img)

            ax[row, col].set_title("True:{} Pred:{:1.3f}".format(Y_val[index], pred)                                                 )

            n += 1
# show mispredictions

display_errors(sorted_errors[0:])
#show more mispredictions

display_errors(sorted_errors[-20:])