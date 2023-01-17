# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import os

import cv2

from tqdm import tqdm
#choosing path for our data. 

DATADIR = "../input"



CATEGORIES = ["images-2d", "images-3d","images-bw","images-real"]

IMAGE_WIDTH = 1000

IMAGE_HIGHT = 500

# preprocessing data 

def preprocessing_images():

    processed_data = [] # contains 2 value 

    for category in CATEGORIES:  # do all classes

        path = os.path.join(DATADIR,category)  # create path to all classes

        class_number = CATEGORIES.index(category)  # get the classification  (0,1,2,3). 0=2d 1=3d, 3= bw, 4=real

        for img in os.listdir(path):  # iterate over each image per classes

            img_data = cv2.imread(os.path.join(path,img),cv2.IMREAD_COLOR 	)  # convert to array

            img_data = cv2.resize(img_data, (IMAGE_WIDTH, IMAGE_HIGHT), cv2.IMREAD_COLOR)

            processed_data.append([img_data,class_number])

    return processed_data

        

        
# getting processed images and save them in a csv file

processed_images = preprocessing_images()

processed_images
import pickle

with open("processed_image_data.txt", "wb") as fp:

    pickle.dump(processed_images, fp )

    

    
#loading the data from txt file

with open("processed_image_data.txt", "rb") as fp:   # Unpickling

        processed_img = pickle.load(fp)

# info of processed_img

# processed_img = [image 1, image 2, .....]

# image 1 = [image1_rgb_value, label] 

# image1_rgb_value = nd array n=3

type(processed_img)

len(processed_images)
import random

random.shuffle(processed_img)

X=[]

Y=[]
for image_data, label in processed_img:

    X.append(image_data)

    Y.append(label)

X= np.array(X)
# Dumping X and Y value in file 

with open("X.pickle", "wb") as fp:

    pickle.dump(X, fp )



with open("Y.pickle", "wb") as fp:

    pickle.dump(Y, fp )

    

#loading the data from txt file

with open("X.pickle", "rb") as fp:   # Unpickling

        X_feature = pickle.load(fp)

        

with open("Y.pickle", "rb") as fp:   # Unpickling

        Y_label = pickle.load(fp)

X_feature.shape[1:]