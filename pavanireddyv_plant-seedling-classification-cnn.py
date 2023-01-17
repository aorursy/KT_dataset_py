from keras.models import Sequential

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.core import Activation, Flatten, Dense

from sklearn.model_selection import GridSearchCV

from keras.layers import BatchNormalization

from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta

from sklearn.model_selection import train_test_split

import numpy as np

import random

import os

import sys

import cv2

from keras.utils import to_categorical

import matplotlib

from subprocess import check_output
def classes_to_int(label):

    # label = classes.index(dir)

    label = label.strip()

    if label == "Black-grass":  return 0

    if label == "Charlock":  return 1

    if label == "Cleavers":  return 2

    if label == "Common Chickweed":  return 3

    if label == "Common wheat":  return 4

    if label == "Fat Hen":  return 5

    if label == "Loose Silky-bent": return 6

    if label == "Maize":  return 7

    if label == "Scentless Mayweed": return 8

    if label == "Shepherds Purse": return 9

    if label == "Small-flowered Cranesbill": return 10

    if label == "Sugar beet": return 11

    print("Invalid Label", label)

    return 12
def int_to_classes(i):

    if i == 0: return "Black-grass"

    elif i == 1: return "Charlock"

    elif i == 2: return "Cleavers"

    elif i == 3: return "Common Chickweed"

    elif i == 4: return "Common wheat"

    elif i == 5: return "Fat Hen"

    elif i == 6: return "Loose Silky-bent"

    elif i == 7: return "Maize"

    elif i == 8: return "Scentless Mayweed"

    elif i == 9: return "Shepherds Purse"

    elif i == 10: return "Small-flowered Cranesbill"

    elif i == 11: return "Sugar beet"

    print("Invalid class ", i)

    return "Invalid Class"
#1.Read the images and generate the train and test dataset (5 points)

NUM_CLASSES = 12

# we need images of same size so we convert them into the size

WIDTH = 128

HEIGHT = 128

DEPTH = 3

inputShape = (WIDTH, HEIGHT, DEPTH)

# initialize number of epochs to train for, initial learning rate and batch size

EPOCHS = 25

INIT_LR = 1e-3

BS = 32



def readTrainData(trainDir):

    data = []

    labels = []

    # loop over the input images

    dirs = os.listdir(trainDir) 

    for dir in dirs:

        absDirPath = os.path.join(os.path.sep,trainDir, dir)

        images = os.listdir(absDirPath)

        for imageFileName in images:

            # load the image, pre-process it, and store it in the data list

            imageFullPath = os.path.join(trainDir, dir, imageFileName)

            #print(imageFullPath)

            img = load_img(imageFullPath)

            arr = img_to_array(img)  # Numpy array with shape (233,233,3)

            arr = cv2.resize(arr, (HEIGHT,WIDTH)) #Numpy array with shape (HEIGHT, WIDTH,3)

            #print(arr.shape) 

            data.append(arr)

            label = classes_to_int(dir)

            labels.append(label)

    return data, labels
def createModel():

    model = Sequential()

    # first set of CONV => RELU => POOL layers

    # The CONV  layer will learn 20 convolution filters, each of which are 5×5.

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))

    #model.add(BatchNormalization()) 

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers

    model.add(Conv2D(64, (3,3), padding="same"))

    #model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3,3), padding="same"))

    #model.add(BatchNormalization())

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=500))

    model.add(Activation("relu"))

    model.add(Dense(units=12))

    model.add(Activation("softmax"))

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    # use binary_crossentropy if there are two classes

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model
#2.Divide the data set into Train and validation data sets

random.seed(10)

allLabels =  os.listdir("../input/train/")  # list of subdirectories and files

print("Loading images...")

sys.stdout.flush()

X, Y = readTrainData("/kaggle/working/../input/train/")

# scale the raw pixel intensities to the range [0, 1]

X = np.array(X, dtype="float") / 255.0

Y = np.array(Y)

# convert the labels from integers to vectors

Y =  to_categorical(Y, num_classes=12)



print("Parttition data into 75:25...")

sys.stdout.flush()

# partition the data into training and testing splits using 75% training and 25% for validation

(trainX, valX, trainY, valY) = train_test_split(X,Y,test_size=0.25, random_state=10)

#3. Initialize & build the model (10 points)

#construct the image generator for data augmentation

print("Generating images...")

sys.stdout.flush()

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \

    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,\

    horizontal_flip=True, fill_mode="nearest")

# initialize the model

print("compiling model...")

sys.stdout.flush()

model = createModel()

# train the network

print("training network...")

sys.stdout.flush()

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), \

    validation_data=(valX, valY), \

    steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)



# save the model to disk

print("Saving model to disk")

sys.stdout.flush()

model.save("/tmp/CNNmodel2")
H.history
# set the matplotlib backend so figures can be saved in the background

# plot the training loss and accuracy

%matplotlib inline

print("Generating plots...")

sys.stdout.flush()

matplotlib.use("Agg")

matplotlib.pyplot.style.use("ggplot")

matplotlib.pyplot.figure()

N = EPOCHS

matplotlib.pyplot.plot(np.arange(0, N), H.history["loss"], label="train_loss")

matplotlib.pyplot.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

matplotlib.pyplot.plot(np.arange(0, N), H.history["acc"], label="train_acc")

matplotlib.pyplot.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")

matplotlib.pyplot.title("Training Loss and Accuracy on  crop classification")

matplotlib.pyplot.xlabel("Epoch #")

matplotlib.pyplot.ylabel("Loss/Accuracy")

matplotlib.pyplot.legend(loc="lower left")

matplotlib.pyplot.savefig("plot.png")

matplotlib.pyplot.show()
def readTestData(testDir):

    data = []

    filenames = []

    # loop over the input images

    images = os.listdir(testDir)

    for imageFileName in images:

        # load the image, pre-process it, and store it in the data list

        imageFullPath = os.path.join(testDir, imageFileName)

        #print(imageFullPath)

        img = load_img(imageFullPath)

        arr = img_to_array(img)  # Numpy array with shape (...,..,3)

        arr = cv2.resize(arr, (HEIGHT,WIDTH)) 

        data.append(arr)

        filenames.append(imageFileName)

    return data, filenames

# read test data and find its classification

testX, filenames = readTestData("/kaggle/working/../input/test/")

# scale the raw pixel intensities to the range [0, 1]

testX = np.array(testX, dtype="float") / 255.0



from keras.models import load_model

mymodel = load_model('/tmp/CNNmodel2')

yFit = mymodel.predict(testX, batch_size=10, verbose=1)

print(yFit) 

print(type(yFit)) # numpy.ndarray

print(type(filenames)) # list



import csv  

with open('output.csv', 'w', newline='') as csvfile:

    fieldnames = ['file', 'species']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for index, file in enumerate(filenames):

        classesProbs = yFit[index]

        maxIdx = 0

        maxProb = 0;

        for idx in range(0,11):

            if(classesProbs[idx] > maxProb):

                maxIdx = idx

                maxProb = classesProbs[idx]

        writer.writerow({'file': file, 'species': int_to_classes(maxIdx)})

print("Writing complete")