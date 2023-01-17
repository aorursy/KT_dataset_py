import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import os

import shutil

import random

import cv2

import matplotlib.pyplot as plt

import scipy.stats

import tensorflow as tf



from keras import applications, optimizers, Input

from keras.models import Sequential, Model

from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.utils.np_utils import to_categorical

from keras.layers.normalization import BatchNormalization



%matplotlib inline
import tarfile



image_list = []

label_list = []



tar = tarfile.open('../input/classification_dataset.tar.gz', "r:gz")

for tarinfo in tar:

    tar.extract(tarinfo.name)

    if(tarinfo.name[-4:] == '.jpg'):

        image_list.append(np.array(cv2.imread(tarinfo.name, cv2.IMREAD_COLOR)))

        label_list.append(tarinfo.name.split('_')[0])

    if(tarinfo.isdir()):

        os.rmdir(tarinfo.name)

    else:

        os.remove(tarinfo.name)    

   

tar.close()



images = np.array(image_list)

labels = np.array(label_list)
def extract_coins(img, to_size=100):

    """

    Find coins on the image and return array

    with all coins in (to_size, to_size) frame 

    

    return (n, to_size, to_size, 3) array

           array of radiuses fo coins

    n - number of coins

    color map: BGR

    """

    # Convert to b&w

    cimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find circles on the image

    circles = cv2.HoughCircles(

        cimg, cv2.HOUGH_GRADIENT, 2, 60, param1=300, param2=30, minRadius=30, maxRadius=50)

    

    # Convert to HSV colorspace

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color range for masking

    lower = np.array([0,0,0])

    upper = np.array([255,255,90])

    # Apply the mask

    mask = cv2.blur(cv2.inRange(hsv, lower, upper), (8, 8))

    

    frames = []

    radiuses = []

    # If circles were not found

    if circles is None:

        return None, None

    

    for circle in circles[0]:

        

        center_x = int(circle[0])

        center_y = int(circle[1])

        

        # If center of coin lays in masked coin range

        if not mask[center_y, center_x]:

            continue

        

        # increase radius by C

        # circle detector tends to decrease radius

        radius = circle[2] + 3

        

        radiuses.append(radius)

        

        # Coordinates of upper left corner of square

        x = int(center_x - radius)

        y = int(center_y - radius)

        

        # As radius was increased the coordinates

        # could go out of bounds

        if y < 0:

            y = 0

        if x < 0:

            x = 0

        

        # Scale coins to the same size

        resized = cv2.resize(img[y: int(y + 2 * radius), x: int(x + 2 * radius)], 

                             (to_size, to_size), 

                             interpolation = cv2.INTER_CUBIC)



        frames.append(resized)



    return np.array(frames), radiuses
# The coins images are extracted from original images using extract_coins function

scaled = []

scaled_labels = []

for nominal, image in zip(labels, images):

    #print(image)

    prepared, _ = extract_coins(image)

    if prepared is not None and len(prepared):

        scaled.append(prepared[0])

        scaled_labels.append(nominal)
# Convert the string labels to int

print(np.array(scaled_labels).shape)

print(set(scaled_labels))

label_classes = set(scaled_labels)



labels_dict = {}

for v_i, v in enumerate(label_classes):

    labels_dict[v] = v_i

        

print(labels_dict)



labels = []

for label in scaled_labels:

    labels.append(labels_dict[label])

    

print(set(labels))    



y_binary = to_categorical(labels)



print(y_binary)
# Plot sample images from list

for image_item in random.sample(scaled, 3): 

    image = np.array(image_item)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.show()
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu')) 

model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.75))          # 0.5

model.add(Dense(5))               # 5 is the number of classes

model.add(Activation('softmax'))



model.compile(

    loss='categorical_crossentropy', 

    optimizer='adam',              # adadelta

    metrics=['accuracy']

)
model.fit(

    x=np.array(scaled),

    y=y_binary,

    epochs=10,

    validation_split=0.15,

    batch_size=500,

    verbose=1                  # 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.

)
image_list = random.sample(scaled, 1)

prediction = model.predict(np.array(image_list))[0]

#print(labels_dict)

#print(prediction)

#print(np.argmax(prediction))

print(

    list(labels_dict.keys())[list(labels_dict.values()).index(np.argmax(prediction))], 

    "{0:.0f}%".format(np.max(prediction) * 100)

)

plt.imshow(cv2.cvtColor(np.array(image_list)[0], cv2.COLOR_BGR2RGB))

plt.show()

    