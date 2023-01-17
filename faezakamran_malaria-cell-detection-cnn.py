#CNN Malaria Detector

# Done by: Mohamed R. Alremeithi





#Importing Modules

import numpy as np # Used for math

import pandas as pd # Used for Data processing

import matplotlib as plt # Used for creating and plotting graphs

import os # Used to open the datasets

import random # Used for random function

from keras.models import Sequential # It imports a

from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization # Used to create layers

import cv2 

from sklearn.model_selection import train_test_split





print(os.listdir("../input/cell_images/cell_images/"))# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Data = []



Uninfected = os.listdir("../input/cell_images/cell_images/Uninfected")

Parasitized = os.listdir("../input/cell_images/cell_images/Parasitized")



for x in Uninfected: # For every uninfected Picture

    Data.append(["../input/cell_images/cell_images/Uninfected/"+x,0]) # Take the Uninfected cell and label it as uninfected(as 0)

    

for x in Parasitized: #For every infected Picture

    Data.append(["../input/cell_images/cell_images/Parasitized/"+x,1]) # Take the Infected cells and label it as infected

    

    

random.shuffle(Data) # Shuffle the datasets to prepare for training





Image = [x[0] for x in Data] # Includes all Imagees 

Label = [x[1] for x in Data] # Includes all Labels (Order of labels match with the images)



del Data

X_train, X_test, Y_train, Y_test = train_test_split(Image, Label, test_size=0.1, random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=46)


def GetPic(path):

    im = cv2.imread(path,1)

    im = cv2.resize(im,(60,60)) # The image will be in 60x60 pixels #* Image would change to 28*28 and 100*100 for testing purposes

    im = im/255

    return im



X_images = []

Y_images = []

X_val_im = []

Y_val_im = []



c = 0



for x in range(len(X_train)):



    try:

        X_images.append(GetPic(X_train[x]))

        Y_images.append(Y_train[x])

        c += 1

    

    except:

        print('c: ' + str(c))



        

Y_train = Y_images





c = 0



for x in range(len(X_val)): #Loop to have val images to X_val_im and Y_val_im



    try:

        X_val_im.append(GetPic(X_val[x]))

        Y_val_im.append(Y_val[x])

    

    except:

        print('c: ' + str(c))

        

Y_val = Y_val_im # part of the validation data





X_images = np.array(X_images)

X_val_im = np.array(X_val_im)
CNN = Sequential() # Creates a new model in which we could add layers 



CNN.add(Conv2D(32, kernel_size=3, activation='relu',input_shape=(60,60,3)))

# CNN.add(Dropout(0.05)) - used for all dropouts

CNN.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))



CNN.add(Conv2D(32, kernel_size=3, activation='relu'))

# CNN.add(Dropout(0.05)) - Used for normal dropout and all dropouts trial

CNN.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

# CNN.add(BatchNormalization()) - Used for the batch normalization trial



CNN.add(Conv2D(16, kernel_size=3, activation='relu')) #Extra Layer

CNN.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))





CNN.add(Flatten())

# CNN.add(Dropout(0.5)) #Used for all dropouts

CNN.add(Dense(1, activation='sigmoid'))



CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','mse','mae'])



History = CNN.fit(X_images, Y_train, validation_data=(X_val_im, Y_val), epochs = 10)

print(CNN.summary())