# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
cwd = os.getcwd()
os.chdir(cwd)
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
trainCats = []
for i in os.listdir("../input/training_set/training_set/cats/"):
    if '(' not in i and '_' not in i:
        i = "../input/training_set/training_set/cats/" + i
        trainCats.append(i)
trainDogs = []
for i in os.listdir("../input/training_set/training_set/dogs/"):
    if '_' not in i and '(' not in i:
        i = "../input/training_set/training_set/dogs/" + i
        trainDogs.append(i)
testCats = []
for i in os.listdir("../input/test_set/test_set/cats/"):
    if '(' not in i and '_' not in i:
        i = "../input/test_set/test_set/cats/" + i
        testCats.append(i)
testDogs = []
for i in os.listdir("../input/test_set/test_set/dogs/"):
    if '(' not in i and '_' not in i:
        i = "../input/test_set/test_set/dogs/" + i
        testDogs.append(i)

Cats, Dogs, All = [], [], []
for i, j in zip(trainCats, trainDogs):
#     Cats.append(i)
#     Dogs.append(j)
    All.append(i)
    All.append(j)
for i, j in zip(testCats, testDogs):
#     Cats.append(i)
#     Dogs.append(j)
    All.append(i)
    All.append(j)

# print("Cats", len(Cats))
# print("Dogs", len(Dogs))
print('All', len(All))
# import shutil
# if os.path.exists("dataset"):
#     shutil.rmtree("dataset")
# os.makedirs("dataset/cats")
# os.chdir("dataset")
# os.makedirs("dogs")
# os.chdir(cwd)
# for i in trainCats:
#     shutil.copy(i, "dataset/cats")
# for i in trainDogs:
#     shutil.copy(i, "dataset/dogs")
# for i in testCats:
#     shutil.copy(i, "dataset/cats")
# for i in testDogs:
#     shutil.copy(i, "dataset/dogs")
!pip install imutils
# Import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2
from keras.utils import np_utils
from imutils import build_montages
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
print("Loading images...")
# imagePaths = list(paths.list_images('dataset'))
imagePaths = All
# print(imagePaths)
print("Loaded")
data = []
labels = []

for imagePath in imagePaths:
    # Extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the image, and resize it to be a fixed 100x100 pixels, ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (100, 100))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
# # Converting data into Numpy Array, scaling it to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
# # Reshaping for channel dimension
data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 3))
# Encoding Labels
LE = LabelEncoder()
labels = LE.fit_transform(labels)
# One-hot Encoding
labels = np_utils.to_categorical(labels, 2)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, stratify=labels, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5)) # DROPOUT
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
epochs = 20
ConvNet = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=epochs, verbose=1)
print("Training Complete")

# from keras.models import load_model

# model.save('myModel.h5')
# print("Model Saved")

# model = load_model('myModel.h5')
print("Evaluating the network")
predictions = model.predict(X_test, batch_size=32)
# Confusion Matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=LE.classes_))
# Plot the training loss and accuracy
%matplotlib inline
plt.style.use("ggplot")
plt.figure(figsize=(20, 10))
plt.plot(np.arange(0, epochs), ConvNet.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), ConvNet.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), ConvNet.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), ConvNet.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# plt.savefig(args["plot"])
# Randomly select test images
testImages = np.arange(0, y_test.shape[0])
testImages = np.random.choice(testImages, size=(25,), replace=False)
images = []
for ti in testImages:
    # Grab the current testing image and classify it
    
    image = np.expand_dims(X_test[ti], axis=0)
    preds = model.predict(image)
    j = preds.argmax(axis=1)[0]
    label = LE.classes_[j]

    # Rescale the image into the range [0, 255] and then resize it so
    # We can more easily visualize it
    output = np.array(image[0] * 255, dtype=np.uint8)
    output = cv2.resize(output, (128, 128))

    # Draw the class label on the output image and add it to the set of output images
    if label == 'cats':
        cv2.putText(output, 'Cat', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(output, 'Dog', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    images.append(output)
# Create a montage using 128x128 "tiles" with 5 rows and 5 columns
montage = build_montages(images, (128, 128), (5, 5))[0]
cv2.imwrite("Output Basic Model.png", montage)
from IPython.display import Image
Image(filename="Output Basic Model.png")
import keras
vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))
vgg16_model.layers.pop()
VGG16model = Sequential(vgg16_model.layers)
for layer in VGG16model.layers:
    layer.trainable = False
VGG16model.add(Flatten())
VGG16model.add(Dense(4096, activation="relu"))
VGG16model.add(Dense(4096, activation="relu"))
VGG16model.add(Dense(2, activation='softmax'))
VGG16model.summary()
VGG16model.compile(Adam(lr=0.0001), loss = 'categorical_crossentropy', metrics=['accuracy'])
epochs = 5
VGG16Net = VGG16model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=epochs, verbose=1)
print("Training Complete")

# from keras.models import load_model

# model.save('VGG16Model.h5')
# print("Model Saved")

print("Evaluating the network")
predictions = VGG16model.predict(X_test, batch_size=32)
# Confusion Matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=LE.classes_))
# Plot the training loss and accuracy
%matplotlib inline
plt.style.use("ggplot")
plt.figure(figsize=(20, 10))
plt.plot(np.arange(0, epochs), VGG16Net.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), VGG16Net.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), VGG16Net.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), VGG16Net.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# plt.savefig(args["plot"])
# Randomly select test images
testImages = np.arange(0, y_test.shape[0])
testImages = np.random.choice(testImages, size=(25,), replace=False)
images = []
for ti in testImages:
    # Grab the current testing image and classify it
    
    image = np.expand_dims(X_test[ti], axis=0)
    preds = VGG16model.predict(image)
    j = preds.argmax(axis=1)[0]
    label = LE.classes_[j]

    # Rescale the image into the range [0, 255] and then resize it so
    # We can more easily visualize it
    output = np.array(image[0] * 255, dtype=np.uint8)
    output = cv2.resize(output, (128, 128))

    # Draw the class label on the output image and add it to the set of output images
    if label == 'cats':
        cv2.putText(output, 'Cat', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(output, 'Dog', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    images.append(output)
# Create a montage using 128x128 "tiles" with 5 rows and 5 columns
montage = build_montages(images, (128, 128), (5, 5))[0]
cv2.imwrite("Output VGG16.png", montage)
from IPython.display import Image
Image(filename="Output VGG16.png")
from keras.layers import GlobalAveragePooling2D

classifier = Sequential()

# Add 2 convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(100, 100, 3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add global average pooling layer
classifier.add(GlobalAveragePooling2D())

# Add full connection
classifier.add(Dense(units=2, activation='softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 30
ThirdModel = classifier.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=epochs, verbose=1)
print("Training Complete")

# from keras.models import load_model

# model.save('ThirdModel.h5')
# print("Model Saved")
print("Evaluating the network")
predictions = classifier.predict(X_test, batch_size=32)
# Confusion Matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=LE.classes_))
# Plot the training loss and accuracy
%matplotlib inline
plt.style.use("ggplot")
plt.figure(figsize=(20, 10))
plt.plot(np.arange(0, epochs), ThirdModel.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), ThirdModel.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), ThirdModel.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), ThirdModel.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# plt.savefig(args["plot"])
# Randomly select test images
testImages = np.arange(0, y_test.shape[0])
testImages = np.random.choice(testImages, size=(25,), replace=False)
images = []
for ti in testImages:
    # Grab the current testing image and classify it
    
    image = np.expand_dims(X_test[ti], axis=0)
    preds = classifier.predict(image)
    j = preds.argmax(axis=1)[0]
    label = LE.classes_[j]

    # Rescale the image into the range [0, 255] and then resize it so
    # We can more easily visualize it
    output = np.array(image[0] * 255, dtype=np.uint8)
    output = cv2.resize(output, (128, 128))

    # Draw the class label on the output image and add it to the set of output images
    if label == 'cats':
        cv2.putText(output, 'Cat', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(output, 'Dog', (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    images.append(output)
# Create a montage using 128x128 "tiles" with 5 rows and 5 columns
montage = build_montages(images, (128, 128), (5, 5))[0]
cv2.imwrite("Output Third Model.png", montage)
from IPython.display import Image
Image(filename="Output Third Model.png")
