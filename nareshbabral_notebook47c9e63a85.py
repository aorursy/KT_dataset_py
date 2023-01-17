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
import pip



def install(package):

    pip.main(['install', package])

    

install("keras")
import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

import skimage.feature

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

import keras

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, Cropping2D

from keras.utils import np_utils

%matplotlib inline
classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]



file_names = os.listdir("../input/Train/")

print(file_names)

file_names = sorted(file_names, key=lambda

                   item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))



file_names = file_names[0:1]

print(file_names)

coordinated_df = pd.DataFrame(index = file_names, columns = classes)

coordinated_df.head()
for filename in file_names:

    image_1 = cv2.imread("../input/TrainDotted/" + filename)

    image_2 = cv2.imread("../input/Train/" + filename)

    #print(image_1)

    #taking absolute difference

    image_3 = cv2.absdiff(image_1, image_2)

    #print(image_3)

    

    #mask out blackened regions in both the images

    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

    mask_1[mask_1 < 20] = 0

    mask_1[mask_1 > 0] = 255

    #print(mask_1)

    

    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    mask_2[mask_2 < 20] = 0

    mask_2[mask_2 > 0] = 255

    #print(type(mask_1))

    

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2)

    

    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)

    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)

    

    adult_males = []

    subadult_males = []

    pups = []

    juveniles = []

    adult_females = []

    #print(blobs[0:5])

    for blob in blobs:

        #finding coordinates for each blob

        y, x, s = blob

        #getting color of image from the original imgae with coordinates

        g, b, r = image_1[int(y)][int(x)][:]

        # decision tree to pick the class of the blob by looking at the color in Train Dotted

        if r > 200 and g < 50 and b < 50: # RED

            adult_males.append((int(x),int(y)))        

        elif r > 200 and g > 200 and b < 50: # MAGENTA

            subadult_males.append((int(x),int(y)))         

        elif r < 100 and g < 100 and 150 < b < 200: # GREEN

            pups.append((int(x),int(y)))

        elif r < 100 and  100 < g and b < 100: # BLUE

            juveniles.append((int(x),int(y))) 

        elif r < 150 and g < 50 and b < 100:  # BROWN

            adult_females.append((int(x),int(y)))

            

    coordinated_df['adult_males'][filename] = adult_males

    coordinated_df['adult_females'][filename] = adult_females

    coordinated_df['pups'][filename] = adult_males

    coordinated_df['juveniles'][filename] = adult_males

    coordinated_df['subadult_males'][filename] = adult_males

coordinated_df.head()