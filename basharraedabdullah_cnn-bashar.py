# Other Userfull Imports. 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

import os



# To deal with files paths(directories) using different notations like (*, _, etc).

import glob



# OpenCV Which deals with image processing.

import cv2



# Import Keras Modules.

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers import LSTM, Input, TimeDistributed

from keras.models import Model

from keras.optimizers import RMSprop, SGD

from keras import backend as K
print(os.listdir("../input/fruits/fruits-360_dataset")) # Print all folders available in the input directory.

print(os.listdir("../input/fruits/fruits-360_dataset/fruits-360")) # Print all folders available in the input directory.

fruit_images = [] # Array where images will be added. 

labels = [] # Lables that are tagged to the pictures.



for fruit_dir_path in glob.glob("../input/fruits/fruits-360_dataset/fruits-360/Training/*"):

      #print(fruit_dir_path) # Returns "../input/fruits-360/Training/filename"

    

    fruit_label = fruit_dir_path.split("/")[-1] # Get the last element in the array which is the lable indicated on the images.

    

    #print(fruit_label) # Will have label names from file name it self

    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):

        #print(image_path) # Will loop on each image in each folder

        image = cv2.imread(image_path, cv2.IMREAD_COLOR) # (imagePath, imageColorMode)

        #cv2.imshow("Original image", image) # Show image

       

        #print(image.shape) # See image dimensions -> (100, 100, 3)

        image = cv2.resize(image, (45, 45)) # Change image dimensions from 100x100 to 45x45

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Change image color mode from RGB to BGR, since openCV deals with BGR.

        #print(image.shape) # See image dimensions after resize -> (45, 45, 3)

        

        fruit_images.append(image)

        labels.append(fruit_label)

        

fruit_images = np.array(fruit_images) # Transform fruit_images array to numpy array for parallel compuation (Increase performance)

labels = np.array(labels)



print('Fruit Images Array:', fruit_images, '\n')

print('Fruit Images Labels:', labels, '\n')
label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))} # Give each lable an index starting from 0 and generate from that a dictionary

id_to_label_dict = {v: k for k, v in label_to_id_dict.items()} # Loop on the previously created dictionary and put it as key-value pairs as (index, lable) format



# Print Key-value pairs

print('Labels:Index - Dictionary Format:', label_to_id_dict, '\n') 

print('Index:Labels - Dictionary Format:', id_to_label_dict, '\n') 

label_ids = np.array([label_to_id_dict[label] for label in labels])



# Print all lable ids

print(label_ids)



# Print images shape, label_id array shape, and labels array shape (Row, Col)

print("fruit_images Shape: ", fruit_images.shape, '\n', 'label_ids Shape:', label_ids.shape, '\n','labels Shape:', labels.shape, '\n') 
# Same operatins which were done on the training data is done here on the validation data.

validation_fruit_images = []

validation_labels = [] 

for fruit_dir_path in glob.glob("../input/fruits/fruits-360_dataset/fruits-360/Test/*"):

    fruit_label = fruit_dir_path.split("/")[-1]

    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (45, 45))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        validation_fruit_images.append(image)

        validation_labels.append(fruit_label)

validation_fruit_images = np.array(validation_fruit_images)

validation_labels = np.array(validation_labels)
validation_label_ids = np.array([label_to_id_dict[label] for label in validation_labels])



# Print all lable ids

print(validation_label_ids)



# Print images shape, label_id array shape, and labels array shape (Row, Col)

print("fruit_images Shape: ", validation_fruit_images.shape, '\n', 'label_ids Shape:', validation_label_ids.shape, '\n') 
# Training Data

X_train = fruit_images

Y_train = label_ids



# Validation Data

X_test = validation_fruit_images

Y_test = validation_label_ids



#print(X_train) # Print each image as a matrix of pixel values (0-255).



#Normalize color values to become between 0 and 1 each pixel has values in this example between (0-255)RGB_Colors we want to make them between 0 & 1.

X_train = X_train/255

X_test = X_test/255



#print(X_train) # Print each image as a matrix of pixel values (0-1).



# Make a flattened version for training and testing.

X_flat_train = X_train.reshape(X_train.shape[0], 45*45*3)

X_flat_test = X_test.reshape(X_test.shape[0], 45*45*3)



# Encode the output to one of the 60 categories we have.

Y_train = keras.utils.to_categorical(Y_train, 120)

Y_test = keras.utils.to_categorical(Y_test, 120)



print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

print('Flattened:', X_flat_train.shape, X_flat_test.shape)
# Print image dimensions and them plot it.

print(X_train[0].shape)

plt.imshow(X_train[0])

plt.show()
   

model_cnn = Sequential()

# First convolutional layer, note the specification of shape

model_cnn.add(Conv2D(32, kernel_size=(3, 3),

                activation='relu',

                input_shape=(45, 45, 3)))

model_cnn.add(Conv2D(64, (3, 3), activation='relu'))

model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Dropout(0.25))

model_cnn.add(Flatten())

model_cnn.add(Dense(128, activation='relu'))

model_cnn.add(Dropout(0.5))

model_cnn.add(Dense(120, activation='softmax')) # Percentage output is converted so that their sum = 1



model_cnn.compile(loss=keras.losses.categorical_crossentropy, # Loss is depending on category not on value such as 2 != 1

# loss = summation(ydesired*log(yactual))        

                optimizer=keras.optimizers.Adadelta(),

                metrics=['accuracy'])

    



model_cnn.fit(X_train, Y_train,

     batch_size=128,

     epochs=1,

     verbose=1,

     validation_data=(X_test, Y_test))

score = model_cnn.evaluate(X_test, Y_test, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])