from os import listdir

from os.path import isfile, join



mypath = "../input/cats-dogs/datasets/images/"

file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(str(len(file_names)) + " images loaded")
import cv2 as cv

import numpy as np

import sys

import os

import shutil



dog_count = 0

cat_count = 0

training_size = 1000

test_size = 500

training_images = []

training_labels = []

test_images = []

test_labels = []

size = 150

dog_dir_train = "./datasets/catsvsdogs/train/dogs/"

cat_dir_train = "./datasets/catsvsdogs/train/cats/"

dog_dir_val = "./datasets/catsvsdogs/validation/dogs/"

cat_dir_val = "./datasets/catsvsdogs/validation/cats/"



def make_dir(directory):

    if os.path.exists(directory):

        shutil.rmtree(directory)

    os.makedirs(directory)

    

make_dir(dog_dir_train)

make_dir(cat_dir_train)

make_dir(dog_dir_val)

make_dir(cat_dir_val)



def getZeros(number):

    if number > 10 and number < 100:

        return "0"

    elif number < 10:

        return "00"

    else:

        return ""

    

for i, file in enumerate(file_names):

    if file_names[i][0] == "d":

        dog_count += 1

        image = cv.imread(mypath + file)

        image = cv.resize(image, (size, size), interpolation=cv.INTER_AREA)

        

        if dog_count <= training_size:

            training_images.append(image)

            training_labels.append(1)

            zeros = getZeros(dog_count)

            cv.imwrite(dog_dir_train + "dog" + str(zeros) + str(dog_count) + ".jpg", image)

        elif dog_count > training_size and dog_count <= training_size + test_size:

            test_images.append(image)

            test_labels.append(1)

            zeros = getZeros(dog_count - 1000)

            cv.imwrite(dog_dir_val + "dog" + str(zeros) + str(dog_count - 1000) + ".jpg", image)      

    elif file_names[i][0] == "c":

        cat_count += 1

        image = cv.imread(mypath + file)

        image = cv.resize(image, (size, size), interpolation=cv.INTER_AREA)

        

        if cat_count <= training_size:

            training_images.append(image)

            training_labels.append(0)

            zeros = getZeros(cat_count)

            cv.imwrite(cat_dir_train + "cat" + str(zeros) + str(cat_count) + ".jpg", image)

        elif cat_count > training_size and cat_count <= training_size + test_size:

            test_images.append(image)

            test_labels.append(0)

            zeros = getZeros(cat_count - 1000)

            cv.imwrite(cat_dir_val + "cat" + str(zeros) + str(cat_count - 1000) + ".jpg", image)

            

    if dog_count == training_size + test_size and cat_count == training_size + test_size:

        break



print("Training and Test Data Extraction Complete")
np.savez("cats_vs_dogs_training_data.npz", np.array(training_images))

np.savez("cats_vs_dogs_training_labels.npz", np.array(training_labels))

np.savez("cats_vs_dogs_test_data.npz", np.array(test_images))

np.savez("cats_vs_dogs_test_labels.npz", np.array(test_labels))
def load_data_training_and_test(datasetname):

    npzfile = np.load(datasetname + "_training_data.npz")

    train = npzfile["arr_0"]

    

    npzfile = np.load(datasetname + "_training_labels.npz")

    train_labels = npzfile["arr_0"]

    

    npzfile = np.load(datasetname + "_test_data.npz")

    test = npzfile["arr_0"]

    

    npzfile = np.load(datasetname + "_test_labels.npz")

    test_labels = npzfile["arr_0"]

    

    return (train, train_labels), (test, test_labels)
for i in range(1, 11):

    random = np.random.randint(0, len(training_images))

    if training_labels[random] == 0:

        print(str(i) + " - Cat")

    else:

        print(str(i) + " - Dog")
(x_train, y_train), (x_test, y_test) = load_data_training_and_test("cats_vs_dogs")



x_train = x_train.astype("float32")

x_test = x_test.astype("float32")



x_train /= 255

x_test /= 255



y_train = y_train.reshape(y_train.shape[0], 1)

y_test = y_test.reshape(y_test.shape[0], 1)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import os



batch_size = 16

epochs = 25



img_rows = x_train[0].shape[0]

img_cols = x_train[1].shape[0]

input_shape = (img_rows, img_cols, 3)



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation("sigmoid"))



model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

print(model.summary())
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)



model.save("cats_vs_dogs_v1.h5")



scores = model.evaluate(x_test, y_test, verbose=1)

print("Test Loss:", scores[0])

print("Test Accuracy:", scores[1])
shutil.rmtree("./datasets/")