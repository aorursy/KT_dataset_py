import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#print(check_output(["ls", "../input/regression_sample"]).decode("utf8"))
import os

import shutil



import cv2

import matplotlib.pyplot as plt

import scipy.stats

import tensorflow as tf



from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



from keras import applications, optimizers, Input

from keras.models import Sequential, Model

from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



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
print(images.shape, labels.shape)
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

    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 2, 60, param1=300, param2=30, minRadius=30, maxRadius=50)

    

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
scaled = []

scaled_labels = []

radiuses = []

for nominal, image in zip(labels, images):

    prepared, radius = extract_coins(image)

    if prepared is not None and len(prepared):

        scaled.append(prepared[0])

        scaled_labels.append(nominal)

        radiuses.append(radius[0])



# Create dataframe with data and pickle it

data = pd.DataFrame({'label': scaled_labels, 'radius': radiuses, 'image': scaled})

data.to_pickle('file.pickle')
# Load data

data = pd.read_pickle('file.pickle')
# Radiuses distribution

data.groupby('label').mean().plot.bar()
# Split the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(

    data.radius.values, data.label.values, test_size=0.20, random_state=42)

X_train, X_test = X_train.reshape(-1, 1), X_test.reshape(-1, 1)
clf = SVC()

clf.fit(X_train, y_train)
clf.score(X_test, y_test)