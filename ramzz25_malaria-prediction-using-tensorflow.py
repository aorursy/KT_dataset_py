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
#Importing required libraries and functions



import cv2

import os

import glob

import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, AvgPool2D, Dense, Flatten, Lambda, Dropout

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#Loading images as list of np array



cell_labels = []

cell_images = []



data_pos = []

data_neg = []



#reading infected cells

img_dir_pos = "../input/cell_images/cell_images/Parasitized" # Directory of all images 

data_path_pos = os.path.join(img_dir_pos,'*g')

files_pos = glob.glob(data_path_pos)

for f in files_pos:

    img = cv2.imread(f)

    img = cv2.resize(img,(128,128))

    cell_images.append(img)

    data_pos.append(img)

    cell_labels.append(1)



#reading uninfected cells

img_dir_neg = "../input/cell_images/cell_images/Uninfected" # Directory of all images

data_path_neg = os.path.join(img_dir_neg,'*g')

files_neg = glob.glob(data_path_neg)

for f in files_neg:

    img = cv2.imread(f)

    img = cv2.resize(img,(128,128))

    cell_images.append(img)

    data_neg.append(img)

    cell_labels.append(0)
#Checking length of the dataset



print(len(cell_images))

print(len(cell_labels))
#Visualizing cells

def Plotting_cells(list_of_cells):

    for i in range(9):

        plt.subplot(3,3,i+1)

        plt.imshow(data_pos[i])
Plotting_cells(data_pos)
Plotting_cells(data_neg)
#changing list of images to np array



cell_images = np.asarray(cell_images)

cell_labels = np.asarray(cell_labels)



print(cell_images.shape)

print(cell_labels.shape)
#splitting the data into test data and train data



x_train, x_test, y_train, y_test = train_test_split(cell_images, cell_labels, test_size=0.3, shuffle=True, random_state=25)
#Checking the shapes of input data



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
#Normalising the input



x_train, x_test = x_train / 255.0, x_test / 255.0
#Defining the neural network with CNNs



model = Sequential([

    #Convulutional Layers

    

    Conv2D(64, kernel_size=(3,3), strides=(2,2), activation='relu', input_shape=(128,128,3)),

    MaxPooling2D(pool_size=(3,3), strides=(2,2)),

    Conv2D(32, kernel_size=(5,5), strides=(2,2), activation='relu'),

    MaxPooling2D(pool_size=(3,3), strides=(1,1)),

    Conv2D(16, kernel_size=(3,3), activation='relu'),

    Flatten(),

    

    #Linear Layers

    Dense(256, activation='relu'),

    Dense(64, activation='relu'),

    Dense(2, activation='softmax')

    

])

model.summary()
#setting hyper-params

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.1)
model.evaluate(x_test,y_test)