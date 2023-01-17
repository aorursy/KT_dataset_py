# EE Experiment: CNN Malaria Detector 

# Done by: Mohamed R. Alremeithi 





#Importing Modules

import numpy as np # Used for math

import os # Used to open the datasets

import random # Used for random function

from keras.models import Sequential # It imports a

from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization # Used to create layers

import cv2 

from sklearn.model_selection import train_test_split





print(os.listdir("../input/cell_images/cell_images/"))



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

X_train, X_test, Y_train, Y_test = train_test_split(Image, Label, test_size=0.1, random_state=42) # Splitting the Training and Testing Data through Sci-kit learn

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=46)


def GetPic(path): # A function that reads a picture then returns it ina 60 x 60 x 3 pixel image format 

    im = cv2.imread(path,1)

    im = cv2.resize(im,(60,60))

    im = im/255

    return im



# Setting up the variables to store the datasets

X_images = []

Y_images = []

X_val_im = []

Y_val_im = []



c = 0



for x in range(len(X_train)): # Inserts X_Train and Y_Train to X_images and Y_images respectively



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





X_images = np.array(X_images) # Creates an array of matrixes for the training and validation 

X_val_im = np.array(X_val_im)

ConvNet = Sequential() # Creates a new model in which we could add layers 

# The layers as described in the methodology



ConvNet.add(Conv2D(32, kernel_size=3, activation='relu',input_shape=(60,60,3))) # Convolutional Layer 

# ConvNet.add(Dropout(0.05)) - used for all dropouts

ConvNet.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2))) # Subsampling Layer 



ConvNet.add(Conv2D(32, kernel_size=3, activation='relu'))

# ConvNet.add(Dropout(0.05)) - Used for normal dropout and all dropouts trial

ConvNet.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))

# ConvNet.add(BatchNormalization()) - Used for the batch normalization trial



ConvNet.add(Conv2D(16, kernel_size=3, activation='relu')) #

ConvNet.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))





ConvNet.add(Flatten()) # Fully Connected Layer

# ConvNet.add(Dropout(0.5)) #Used for all dropouts

ConvNet.add(Dense(1, activation='sigmoid')) # Loss Layer



ConvNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','mse','mae']) 



ConvNet.fit(X_images, Y_train, validation_data=(X_val_im, Y_val), epochs = 10) # The Network starts the Learning and Testing Process

print(ConvNet.summary()) # The Structure and Summary of the Network 