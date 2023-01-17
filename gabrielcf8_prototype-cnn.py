#!pip install -U tf-nightly-gpu

import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)

import cv2

import glob

import numpy as np

import os, sys, time

import numpy as numpy

import pandas as pd 

import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



import tqdm

from PIL import Image

from collections import OrderedDict



# The tf.keras and keras, they are not compatible. Choose only one.

#import keras

from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input

from keras.utils.np_utils import to_categorical



from keras.utils import np_utils

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import RMSprop



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
total_df = pd.read_csv("../input/humpback-whale-identification/train.csv")

total_df.head()
print(f"There are {len(os.listdir('../input/humpback-whale-identification/train'))} images in train dataset with {total_df.Id.nunique()} unique classes.")

print(f"There are {len(os.listdir('../input/humpback-whale-identification/test'))} images in test dataset.")
total_df.Id.value_counts().head()
total_df.Id.describe()
def filterForTails (img):

    draw = False



    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(imgray, (7, 7), 0)

    ret,thresh = cv2.threshold(blur,127,255,0)



    blur2 = cv2.GaussianBlur(thresh, (7, 7), 0)

    thresh2 = cv2.threshold(blur2, 250, 250, cv2.THRESH_BINARY)[1]



    contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    for c in contours2:

        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)



        x, y, w, h = cv2.boundingRect(c)

        roi = img[y:h+y, x:w+x]

        imgray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)



        if (len(approx) != 1):

            draw = True



    if (draw):

        img = cv2.drawContours(img, contours2, -3, (255, 255, 255), -8)

        draw = False



    return img
def prepareImages(train, shape, image_size, channel, path, filterOn):

    dt_proc = np.zeros((shape, image_size, image_size, channel))

    count = 0

    u = True

    for fig in train['Image']:

        

        #load images into images of size 100x100x3

        img = cv2.imread("../input/humpback-whale-identification/"+path+"/"+fig)

        if(filterOn):

            img = filterForTails(img)

        img = cv2.resize(img, (100, 100))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        dt_proc[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return dt_proc
image_size = 100

channel = 3



#change to FALSE to not use filter on images

filterOn = True



start = time.time()

dt_proc = prepareImages(total_df, total_df.shape[0], image_size, channel, "train", filterOn)

end = time.time()



elapsed = end - start

print("Tempo de carregamento imagens: ", elapsed)
dt_proc = dt_proc / 255.0

print("dt_proc shape: ",dt_proc.shape)



IMG_SHAPE = dt_proc[0].shape

print("IMG_SHAPE: ", IMG_SHAPE)
label_encoder = LabelEncoder()
y_total = total_df["Id"]

y_total = label_encoder.fit_transform(y_total)
# convert to one-hot-encoding(one hot vectors)

# we have 5005 class look at from=> train.Id.describe()

y_total = to_categorical(y_total, num_classes = 5005)
#  Shuffle

#x,y = shuffle(x_train,y_train, random_state=2)



X_train, X_test, Y_train, Y_test = train_test_split(dt_proc, y_total, test_size=0.2, random_state=2)

dt_proc = 0

y_total = 0
# # With data augmentation to prevent overfitting



# datagen = ImageDataGenerator(

#         featurewise_center=False,  # set input mean to 0 over the dataset

#         samplewise_center=False,  # set each sample mean to 0

#         featurewise_std_normalization=False,  # divide inputs by std of the dataset

#         samplewise_std_normalization=False,  # divide each input by its std

#         zca_whitening=False,  # apply ZCA whitening

#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

#         zoom_range = 0.1, # Randomly zoom image 

#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

#         horizontal_flip=False,  # randomly flip images

#         vertical_flip=False)  # randomly flip images





# datagen.fit(x_train)
epochs = 10



# Model structure

model = Sequential()



#FIRST BLOCK

model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=IMG_SHAPE))

model.add(Activation('relu'))

model.add(Convolution2D(32, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



#SECOND BLOCK

model.add(Convolution2D(64, 3, 3))

model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



#THIRD BLOCK

model.add(Flatten())

model.add(Dense(32))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(5005))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])



# Model View (run with this is not necessary)

model.summary()



model.get_config()

model.layers[0].get_config()

model.layers[0].input_shape

model.layers[0].output_shape

model.layers[0].get_weights()

np.shape(model.layers[0].get_weights()[0])

model.layers[0].trainable



# Training

start = time.time()

hist = model.fit(X_train, Y_train, batch_size=4, nb_epoch=epochs, verbose=1, validation_data=(X_test, Y_test))

end = time.time()

elapsed = end - start

print("Tempo de treino: ", elapsed)
# Plot the loss curve for training

plt.plot(hist.history['loss'], color='r', label="Train Loss")

plt.title("Train Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Plot the accuracy curve for training

plt.plot(hist.history['accuracy'], color='g', label="Train Accuracy")

plt.title("Train Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
print('Train accuracy of the model: ',hist.history['accuracy'][-1])

print('Train loss of the model: ',hist.history['loss'][-1])
predictions = model.predict(np.array(X_test), verbose=1)
# Evaluating the model

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test Loss:', score[0])

print('Test accuracy:', score[1])





# Plot Acurracy vs Loss values

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['loss'])

plt.title('Model Accuracy vs Loss')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Accuracy (' + str(round(score[1],2)) + ')', 'Loss (' + str(round(score[0],2)) + ')'], loc='upper left')

plt.show()