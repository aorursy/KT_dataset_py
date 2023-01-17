# imports

import os, cv2, itertools, re

import keras

from keras.utils.np_utils import to_categorical

from keras.datasets import mnist

from keras.layers import Convolution2D, Dropout

from keras.models import Sequential

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D

from keras.models import Model

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import random as rnd

from keras import backend as K

from keras.utils import np_utils

from keras.callbacks import CSVLogger

from keras.models import load_model

from keras.utils import np_utils as npu

from keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import shuffle

import sklearn

from sklearn.model_selection import train_test_split



# function to return the most frequent element in a list (for good/bad predictions)

def most_frequent(listt):

    return max(set(listt), key = listt.count)



# the mnist model

def mnist_model():

    model = Sequential()



    model.add(Convolution2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

    model.add(Dropout(0.25))



    model.add(Convolution2D(64, (3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

    model.add(Dropout(0.25))



    model.add(Flatten())



    model.add(Dense(128))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))



    model.add(Dense(10))

    model.add(Activation('softmax'))



    return model



# the cats vs dogs convolutional model

def animals_model():

    model = Sequential()



    model.add(Convolution2D(32, (3,3), padding='same', input_shape=(80, 80, 1)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Convolution2D(64, (3,3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.4))



    model.add(Convolution2D(128, (3,3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.4))



    model.add(Convolution2D(256, (3,3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.4))

    

    model.add(Convolution2D(512, (3,3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.4))



    model.add(Flatten())

    

    model.add(Dropout(0.4))



    model.add(Dense(120))

    model.add(Activation('relu'))

    

    model.add(Dense(2))

    model.add(Activation('softmax'))

    

    return model
# setting the batch size for each epoch, number of classes (2), number of epochs

BATCH_SIZE = 64

num_classes = 2

NUM_EPOCHS = 50



# array of labels for plotting confusion matrix and badly classified images

animal_labels = ["cat", "dog"]



# paths to the train and test data

TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train/'

TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test/'



# data preparation

X = []

y = []

convert = lambda category : int(category == 'dog')

# importing and processing training data

def create_test_data(path):

    for p in os.listdir(path):

        if(os.path.join(path,p) != '../input/dogs-vs-cats-redux-kernels-edition/train/train'):

            category = p.split(".")[0]

            category = convert(category)

            # conversion to grayscale

            img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

            # resize all images to 80x80 pixels

            new_img_array = cv2.resize(img_array, dsize=(80, 80))

            X.append(new_img_array)

            y.append(category)

    

create_test_data(TRAIN_DIR)

X = np.array(X).reshape(-1, 80,80,1)

y = np.array(y)



# normalizing image data

X = X/255.0



# validation and training split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)



# importing the test data

X_test = []

def create_test1_data(path):

    for p in os.listdir(path):

        if(os.path.join(path,p) != '../input/dogs-vs-cats-redux-kernels-edition/test/test'):

            img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)

            new_img_array = cv2.resize(img_array, dsize=(80, 80))

            X_test.append(new_img_array)



create_test1_data(TEST_DIR)

X_test = np.array(X_test).reshape(-1,80,80,1)

X_test = X_test/255



# one hot encoding of vector labels: 0 - cat, 1- dog

y_train = to_categorical(y_train)

y_val = to_categorical(y_val)



print(y_val.shape)



# initializing conv model

animal_cnn = animals_model()



# compiling model

animal_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# imageDataGenerators for data augmentation

train_datagen = ImageDataGenerator(

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



val_datagen = ImageDataGenerator(

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

validation_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)



# training the network

train_history = animal_cnn.fit_generator(

    train_generator, 

    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,

    epochs=NUM_EPOCHS,

    validation_data=validation_generator,

    verbose=0,

    validation_steps=X_val.shape[0] // BATCH_SIZE

)



# saving the network

animal_cnn.save('dogs_cats_model.h5')



# testing network performance

score = animal_cnn.evaluate(X_val, y_val, verbose=1)

print('Dogs vs Cats Evaluation Loss: ', score[0])

print('Dogs vs Cats Evaluation Accuracy: ', score[1])



# Print Model Stats

print('Training accuracy')

print(max(train_history.history['accuracy']))



print('Validation accuracy')

print(max(train_history.history['val_accuracy']))



# plotting training/validation accuracy/loss evolution

plt.figure()

plt.plot(train_history.history['accuracy'], color='C0', label='Training acc')

plt.title('Dogs vs Cats CNN Training Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(loc='lower right')

plt.show()



plt.figure()

plt.plot(train_history.history['loss'], color='C0', label='Training loss')

plt.title('Dogs vs Cats CNN Training Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(loc='upper right')

plt.show()



plt.figure()

plt.plot(train_history.history['val_accuracy'], color='C0', label='Validation acc')

plt.title('Dogs vs Cats CNN Validation Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(loc='lower right')

plt.show()



plt.figure()

plt.plot(train_history.history['val_loss'], color='C0', label='Validation loss')

plt.title('Dogs vs Cats CNN Validation Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(loc='upper right')

plt.show()



# getting predictions for the test data for the confusion matrix

y_pred = animal_cnn.predict(X_val)

y_pred = np.argmax(y_pred,axis=1)

y_true = np.asarray([np.argmax(i) for i in y_val])

print(y_pred.shape)

print(y_true.shape)



# generating confusion matrix

cm = confusion_matrix(y_true, y_pred)

print(cm)

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



# plotting confusion matrix

sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(10,10))

ax = sns.heatmap(

    cm_norm, annot=True, linewidths=0, square=False, cmap='Greens',

    yticklabels=animal_labels, xticklabels=animal_labels,

    vmin=0, vmax=np.max(cm_norm), fmt='.2f',

    annot_kws={'size': 20}

)

ax.set(xlabel='Predicted Label', ylabel='Actual Label', title='Dogs vs Cats CNN Confusion Matrix')

plt.show()



# plotting predictions for the first 10 test images

for x in range(0, 10):

    original_test_image = X_test[x].reshape(1,80, 80, 1)

    prediction = animal_cnn.predict(original_test_image)

    prediction_text = ""

    pred_val = max(prediction[0])

    if prediction[0][0] == pred_val:

        prediction_text = "cat"

    else:

        prediction_text = "dog"

    

    plt.xticks([])

    plt.yticks([])

    plt.imshow(X_test[x, :, :, 0], cmap='gray')

    title_text = "Predicted: " + prediction_text

    pic_name = "prediction" + str(x) + ".jpg"

    plt.title(title_text, fontsize=20)

    plt.savefig(pic_name)

    plt.show()