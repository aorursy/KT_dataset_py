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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#First, I need to organize my data in a way that seperates it into categories. This data set, fortunately, is already

#seperated into test and valid data. I just need to orient my data to where each one is what it is labeled as what it is



#This notebook took a bit of getting used to 



import cv2

import os

import random

import matplotlib.pylab as plt

from glob import glob

import pandas as pd

import numpy as np





for (root,dirs,files) in os.walk('/kaggle/input'): 

        print (root) 

        print (dirs)

        print (files) 

        print ('--------------------------------')



os.path.isdir("/kaggle/input/genderdetectionface/dataset1/dataset1")

input_directory = "../input/"



#With this code, I have now attempted to separate all my data into groups based on what I have labeled them



for valid_man_images in os.path.dirname("/kaggle/input/genderdetectionface/dataset1/valid/man/face_219.jpg"):

    valid_man_images = os.listdir("/kaggle/input/genderdetectionface/dataset1/valid/man")

    print (valid_man_images)

    

for valid_woman_images in os.path.dirname("/kaggle/input/genderdetectionface/dataset1/valid/woman/face_354.jpg"):

    valid_woman_images = os.listdir("/kaggle/input/genderdetectionface/dataset1/valid/woman")

    print (valid_woman_images)



for train_man_images in os.path.dirname("/kaggle/input/genderdetectionface/dataset1/train/man/face_1261.jpg"):

    train_man_images = os.listdir("/kaggle/input/genderdetectionface/dataset1/train/man")

    print (train_man_images)

    

for train_woman_images in os.path.dirname("/kaggle/input/genderdetectionface/dataset1/train/woman/face_1083.jpg"):

    train_woman_images = os.listdir("/kaggle/input/genderdetectionface/dataset1/train/woman")

    print (train_woman_images)



for test_man_images in os.path.dirname("/kaggle/input/genderdetectionface/dataset1/test/man/face_21.jpg"):

    test_man_images = os.listdir("/kaggle/input/genderdetectionface/dataset1/test/man")

    print (test_man_images)

    

for test_woman_images in os.path.dirname("/kaggle/input/genderdetectionface/dataset1/test/woman/face_151.jpg"):

    test_woman_images = os.listdir("/kaggle/input/genderdetectionface/dataset1/test/woman")

    print (test_woman_images)

 

#Now I want to get the arrays (pixel values) of all of the images. This is because the neural network I will be using later 

#is not able to process images themselves. It can only process numerical inputs 





import numpy as np 

import matplotlib.pyplot as plt

import PIL 

from PIL import Image





valid_man_image_arrays = []

for valid_man_images in os.listdir("/kaggle/input/genderdetectionface/dataset1/dataset1/valid/man"):

    valid_man_images = Image.open("/kaggle/input/genderdetectionface/dataset1/dataset1/valid/man/" + valid_man_images)

    valid_man_images = np.array(valid_man_images)

    valid_man_image_arrays.append(valid_man_images)



valid_woman_image_arrays = []

for valid_woman_images in os.listdir("/kaggle/input/genderdetectionface/dataset1/dataset1/valid/woman"):

    valid_woman_images = Image.open("/kaggle/input/genderdetectionface/dataset1/dataset1/valid/woman/" + valid_woman_images)

    valid_woman_images = np.array(valid_woman_images)

    valid_woman_image_arrays.append(valid_woman_images)



train_man_image_arrays = []

for train_man_images in os.listdir("/kaggle/input/genderdetectionface/dataset1/dataset1/train/man"):

    train_man_images = Image.open("/kaggle/input/genderdetectionface/dataset1/dataset1/train/man/" + train_man_images)

    train_man_images = np.array(train_man_images)

    train_man_image_arrays.append(train_man_images)

    

train_woman_image_arrays = []

for train_woman_images in os.listdir("/kaggle/input/genderdetectionface/dataset1/dataset1/train/woman"):

    train_woman_images = Image.open ("/kaggle/input/genderdetectionface/dataset1/dataset1/train/woman/" + train_woman_images)

    train_woman_images = np.array(train_woman_images)

    train_woman_image_arrays.append(train_woman_images) 



test_man_image_arrays = []

for test_man_images in os.listdir("/kaggle/input/genderdetectionface/dataset1/dataset1/test/man"):

    test_man_images = Image.open("/kaggle/input/genderdetectionface/dataset1/dataset1/test/man/" + test_man_images)

    test_man_images = np.array(test_man_images)

    test_man_image_arrays.append(test_man_images)



test_woman_image_arrays = []

for test_woman_images in os.listdir("/kaggle/input/genderdetectionface/dataset1/dataset1/test/woman"):

    test_woman_images = Image.open("/kaggle/input/genderdetectionface/dataset1/dataset1/test/woman/" + test_woman_images)

    test_woman_images = np.array(test_woman_images)

    test_woman_image_arrays.append(test_woman_images) 



# This is the part where I define a function that converts all of these arrays I have created above, into black and white images



import numpy as np 

import matplotlib.pyplot as plt

import PIL 

from PIL import Image 



#these rgb weights are essential to converting from color to grayscale

rgb_weights =  [0.2989, 0.5870, 0.1140]



def color_conversion(array):

    return np.dot(array[...,:3], rgb_weights)

#Now I got to test my function and see if it displays everything in the way that I want it to be displayed. I have to run my function through every image

test = color_conversion(test_man_image_arrays[100])

test



gray_image = plt.imshow(test, cmap = "Greys")

#np.array(test_man_image_arrays)

#test_man_image_arrays[:,:,:] *= [0.2989, 0.5870, 0.1140]

test_man_image_arrays[0].shape
#Now that I have all of my images into arrays, and have created a function that will turn them into black and white arrays

#I can begin the machine learning 



import pandas as pd 

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)



for images in (train_man_image_arrays):

    train_man = color_conversion(images)



for images in (train_woman_image_arrays):

    train_woman = color_conversion(images)



#First I need to one hot incode my data, and give binary values to man and woman images so that I can input this into my data

train_images = vstack((train_man, train_woman))

print (train_images)












