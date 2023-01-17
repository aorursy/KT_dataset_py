# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os, sys

from IPython.display import display

from IPython.display import Image as _Imgdis

from PIL import Image

import numpy as np

from time import time

from time import sleep
folder = "../input/regression_sample"



onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]



print("Working with {0} images".format(len(onlyfiles)))

print("Image examples: ")



for i in range(40, 42):

    print(onlyfiles[i])

    display(_Imgdis(filename=folder + "/" + onlyfiles[i], width=240, height=320))
from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



train_files = []

y_train = []

i=0

for _file in onlyfiles:

    train_files.append(_file)

    label_in_file = _file.find("_")

    y_train.append(int(_file[0:label_in_file]))

    

print("Files in train_files: %d" % len(train_files))



# Original Dimensions

image_width = 640

image_height = 480

ratio = 4



image_width = int(image_width / ratio)

image_height = int(image_height / ratio)



channels = 3

nb_classes = 1



dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),

                     dtype=np.float32)



i = 0

for _file in train_files:

    img = load_img(folder + "/" + _file)  # this is a PIL image

    img.thumbnail((image_width, image_height))

    # Convert to Numpy Array

    x = img_to_array(img)  

    #x = x.reshape((3, 120, 160))

    # Normalize

    x = (x - 128.0) / 128.0

    dataset[i] = x

    i += 1

    if i % 250 == 0:

        print("%d images to array" % i)

print("All images to array!")
from sklearn.cross_validation import train_test_split

#Splitting 

X_train, X_test, y_train, y_test = train_test_split(dataset, y_train, test_size=0.2, random_state=33)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=33)

print("Train set size: {0}, Val set size: {1}, Test set size: {2}".format(len(X_train), len(X_val), len(X_test)))
y_train = np.array(y_train)

y_val = np.array(y_val)

y_test = np.array(y_test)

y_train[:5]
datagen = ImageDataGenerator(

        rotation_range=90,

        shear_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



datagen.fit(X_train)



val_datagen = ImageDataGenerator(

        rotation_range=90,

        shear_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



val_datagen.fit(X_val)
from keras.optimizers import SGD

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.callbacks import EarlyStopping



model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=X_train[0].shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(32, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(64, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten()) 

model.add(Dense(64))

model.add(Activation('linear'))

model.add(Dropout(0.5))

model.add(Dense(1))



sgd = SGD(lr=0.01, decay=1e-7, momentum=.9)

    

early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')



model.compile(loss='mse', optimizer='rmsprop')



model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), 

                    samples_per_epoch=len(X_train),  

                    nb_epoch=10, 

                    validation_data=val_datagen.flow(X_val, y_val, batch_size=32),

                    nb_val_samples=len(X_val),

                    verbose=1,

                    callbacks=[early_stop])
from math import sqrt

start = time()

predictions = model.predict(X_test)

print("Mean prediction time: {0:.2f}s".format((time() - start)/len(X_test)))

print("Mean prediction error: {0:.2f} cents".format(sqrt(model.evaluate(X_test, y_test))))
print("{0:5} {1:10} {2:8}".format("True", "Predicted", "Error"))

for i in range(len(predictions[:10])):

    print("{0:5} {1:10} {2:8}".format(y_test[i], int(predictions[i]), abs(y_test[i] - int(predictions[i]))))
### Make predictions 5 cents multiples

def closer_multiple(x, mult):

    value = mult * int(x / multi)

    if (x % multi) > multi/2.0:

        value += multi

    return value



multi = 5

predictions = [closer_multiple(x, multi) for x in predictions]



print("{0:5} {1:10} {2:8}".format("True", "Predicted", "Error"))

for i in range(len(predictions[:10])):

    print("{0:5} {1:10} {2:8}".format(y_test[i], int(predictions[i]), abs(y_test[i] - int(predictions[i]))))
error = np.mean(abs(np.array(predictions) - y_test))

print("Mean prediction error using multiples of 5: {0:.2f} cents".format(error))