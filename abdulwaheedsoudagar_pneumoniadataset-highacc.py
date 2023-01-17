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

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K



from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random

import pickle

import cv2

import os

import re

%matplotlib inline

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle
print("[INFO] loading images...")

imagePaths=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        x = re.search("^.*.jpeg$", filename)

        if (x):

            imagePaths.append(dirname+filename)
labels = {'NORMAL': 0, 'PNEUMONIA': 1}

PATH = '/kaggle/input/chest-xray-pneumonia/chest_xray/'
EPOCHS = 20

INIT_LR = 0.0001

BS = 16

IMAGE_DIMS = (96, 96, 1)
def DataSetPrep(dir_,data,label):

   for category in ['NORMAL', 'PNEUMONIA']:

        loc = os.path.join(PATH, dir_, category)

        for name in os.listdir(loc):

            image = cv2.imread(os.path.join(loc, name), 0)

            image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))

            image = img_to_array(image)

            data.append(image)

            label.append(labels[category])

   return data, label
data = []

label = []
data, label = DataSetPrep('train',data,label)

data, label = DataSetPrep('test',data,label)

data, label = DataSetPrep('val',data,label)
data, label = shuffle(data, label, random_state=42)
data = np.array(data, dtype="float") / 255.0

label = np.array(label)
data.shape
#data = data.reshape( -1,IMAGE_DIMS[1], IMAGE_DIMS[0], 3)
print("[INFO] data matrix: {:.2f}MB".format(

	data.nbytes / (1024 * 1000.0)))
lb = LabelBinarizer()

label = lb.fit_transform(label)
data.shape
(trainX, testX, trainY, testY) = train_test_split(data,label, test_size=0.1, random_state=42)

(trainX, X_cv, trainY, y_cv) = train_test_split(trainX,trainY, test_size=0.1, random_state=42)
from keras.utils import to_categorical

trainY = to_categorical(trainY, num_classes=2)

testY = to_categorical(testY, num_classes=2)

y_cv = to_categorical(y_cv, num_classes=2) 
from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K



class CNN_Network:

	@staticmethod

	def build(width, height, depth, classes):

		model = Sequential()

		inputShape = (height, width, depth)

		chanDim = -1



		# if we are using "channels first", update the input shape and channels dimension

		if K.image_data_format() == "channels_first":

			inputShape = (depth, height, width)

			chanDim = 1

		model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(3, 3)))

		model.add(Dropout(0.25))



		model.add(Conv2D(64, (3, 3), padding="same"))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

  

		model.add(Conv2D(64, (3, 3), padding="same"))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Dropout(0.25))



		model.add(Conv2D(128, (3, 3), padding="same"))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

  

		model.add(Conv2D(128, (3, 3), padding="same"))

		model.add(Activation("relu"))

		model.add(BatchNormalization(axis=chanDim))

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Dropout(0.25))



		model.add(Flatten())

		model.add(Dense(1024))

		model.add(Activation("relu"))

		model.add(BatchNormalization())

		model.add(Dropout(0.5))



		model.add(Dense(classes))

		model.add(Activation("softmax"))



		return model
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,

	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,

	horizontal_flip=True, fill_mode="nearest")
print("[INFO] compiling model...")

model = CNN_Network.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],

	depth=IMAGE_DIMS[2], classes=len(lb.classes_))

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt,

	metrics=["accuracy"])
EPOCHS = 20

INIT_LR = 0.00001

BS = 16
# train the network

print("[INFO] training network...")

H = model.fit_generator(

	aug.flow(trainX, trainY, batch_size=BS),

	validation_data=(testX, testY),

	steps_per_epoch=len(trainX) // BS,

	epochs=EPOCHS, verbose=1)


plt.style.use("ggplot")

plt.figure()

N = EPOCHS

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

plt.title("Training Loss")

plt.xlabel("Epoch #")

plt.ylabel("Loss")

plt.legend(loc="upper left")

    
plt.style.use("ggplot")

plt.figure()

N = EPOCHS



plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Accuracy")

plt.xlabel("Epoch #")

plt.ylabel("Accuracy")

plt.legend(loc="upper left")
test_loss, test_score = model.evaluate(X_cv, y_cv)

print("Loss on test set: ", test_loss)

print("Accuracy on test set: ", test_score)
preds = model.predict(X_cv)

preds = np.argmax(preds, axis=-1)



orig_test_labels = np.argmax(y_cv, axis=-1)
cm  = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
tn, fp, fn, tp = cm.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)



print("Recall of the model is {:.2f}".format(recall))

print("Precision of the model is {:.4f}".format(precision))