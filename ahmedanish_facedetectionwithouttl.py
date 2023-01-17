# First let us all all the necessary packages and the training and validation data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

import matplotlib.pyplot as plt

#from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# Lets check for a random image

img = cv2.imread('/kaggle/input/5-celebrity-faces-dataset/data/val/madonna/httpassetsrollingstonecomassetsarticlemadonnadavidbowiechangedthecourseofmylifeforeversmallsquarexmadonnabowiejpg.jpg',12)
# Now we shall display the image

from matplotlib import pyplot as plt

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

plt.show()
# Load training dataset of the faces data

for imgfolder in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"):

  for filename in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"+ imgfolder):

    filename = "/kaggle/input/5-celebrity-faces-dataset/data/train/" + imgfolder + "/" + filename

    print(filename)

    img = cv2.imread(filename,0)

    plt.imshow(img, cmap ='gray', interpolation= 'bicubic' )

    plt.xticks([])

    plt.yticks([])

    plt.show()
#Let us now check for the size of each image

for imgfolder in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"):

  for filename in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"+ imgfolder):

    filename = "/kaggle/input/5-celebrity-faces-dataset/data/train/" + imgfolder + "/" + filename

    img = cv2.imread(filename,0)

    print(img.shape)
#Now we will resize all the images to same format/size

for imgfolder in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"):

  for filename in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"+ imgfolder):

    filename = "/kaggle/input/5-celebrity-faces-dataset/data/train/" + imgfolder + "/" + filename

    img = cv2.imread(filename,0)

    img = cv2.resize(img, (47,62), interpolation = cv2.INTER_AREA)

    print (img.shape)
# Lets load the image data for training

X_images = []

for imgfolder in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"):

  for filename in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"+ imgfolder):

    filename = "/kaggle/input/5-celebrity-faces-dataset/data/train/" + imgfolder + "/" + filename

    img = cv2.imread(filename,0)

    img = cv2.resize(img, (47,62), interpolation = cv2.INTER_AREA)

    X_images.append(img)

X_images = np.asarray(X_images)

print(X_images.shape)
# Lets look at the image after resizing

plt.imshow(X_images[34],cmap = 'gray', interpolation='bicubic')
#lets load the output data for training

y_train = []

for imgfolder in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"):

  for filename in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/train/"+ imgfolder):

    filename = "/kaggle/input/5-celebrity-faces-dataset/data/train/" + imgfolder + "/" + filename

    y_train.append(imgfolder)

y_train = np.asarray(y_train)

print (y_train.shape)
#lets load val/testing data 

X_test = []

for imgfolder in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/val/"):

  for filename in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/val/"+ imgfolder):

    filename = "/kaggle/input/5-celebrity-faces-dataset/data/val/" + imgfolder + "/" + filename

    img = cv2.imread(filename,0)

    img = cv2.resize(img, (47,62), interpolation = cv2.INTER_AREA)

    X_test.append(img)

X_test = np.asarray(X_test)

print(X_test.shape)
# now we will load output testing data

y_test = []

for imgfolder in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/val/"):

  for filename in os.listdir("/kaggle/input/5-celebrity-faces-dataset/data/val/"+ imgfolder):

    filename = "/kaggle/input/5-celebrity-faces-dataset/data/val/" + imgfolder + "/" + filename

    y_test.append(imgfolder)

y_test = np.asarray(y_test)

print (y_test.shape)
#we will encode the output data to some integer value using label encoder

from sklearn.preprocessing import LabelEncoder

labenc = LabelEncoder()

y_train = labenc.fit_transform(y_train)

y_test = labenc.fit_transform(y_test)
#Now let us model our trainig network using keras and also add neccessry hidden layers

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (62,47)),

                                    tf.keras.layers.Dense(1024,activation='relu'),

                                    tf.keras.layers.Dense(256,activation='relu'),

                                    tf.keras.layers.Dropout(0.2),

                                    tf.keras.layers.Dense(5,activation='softmax')])

model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy'] )
# lastly we will fit our model on our training data and evaluate the model accuracy

model.fit(X_images, y_train,epochs=40)



model.evaluate(X_test, y_test,verbose =2)