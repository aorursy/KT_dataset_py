# importing required libraries

import numpy as np 

import pandas as pd

import csv

from PIL import Image

import string
# storing the location of the csv file

File_Path ='../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data/A_Z Handwritten Data.csv'
#reading the data using pandas

data = pd.read_csv('../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data/A_Z Handwritten Data.csv')

# separating the independent and dependent variable 

X = data.drop('0',axis = 1)

y = data['0']
#splitting the data into training and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print(X_train.shape)

print(X_test.shape)
# creating a dataframe of training images and labels

train_images_csv = pd.concat([y_train, X_train], axis =1)

# sorting the dataframe lexicographically

train_images_csv = train_images_csv.sort_values('0')



# similarly a dataframe of testing images and labels and sorting.

test_images_csv = pd.concat([y_test, X_test], axis =1)

test_images_csv = test_images_csv.sort_values('0')



# storing the training and testing data frame as csv

export_csv_training = train_images_csv.to_csv ('/kaggle/training.csv', index = None)

export_csv_test = test_images_csv.to_csv ('/kaggle/test.csv', index = None)
# This section reads the above created csv and converts and stores them as images in different folders 

image_Folder_Path = '/kaggle/image'

image_Folder_Path_training = "/kaggle/image/training-set"

image_Folder_Path_test = "/kaggle/image/test-set"



Alphabet_Mapping_List = list(string.ascii_uppercase)



import os



# creates folders with alphabets name where training images will be stored

for alphabet in Alphabet_Mapping_List:

    path = image_Folder_Path_training + '/' + alphabet

    if not os.path.exists(path):

        os.makedirs(path)

        

# creates folders with alphabets name where testing images will be stored

for alphabet in Alphabet_Mapping_List:

    path = image_Folder_Path_test + '/' + alphabet

    if not os.path.exists(path):

        os.makedirs(path)

# The below code converts csv to images.



training_csv = '/kaggle/training.csv'

test_csv ='/kaggle/test.csv'

#The below code should be executed once for creating the training image and once for testing image

def convert_csv_to_image (filePath,image_Folder_Path):

    count = 1

    last_digit_Name =  None

    with open(filePath,newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        count = 0

        for row in reader:

            digit_Name = row[0]

            image_array = np.asarray(row[1:])

            image_array = image_array.reshape(28, 28)

            new_image = Image.fromarray(image_array.astype('float'))

            new_image = new_image.convert("L")



            if last_digit_Name != str(Alphabet_Mapping_List[(int)(digit_Name)]):

                last_digit_Name = str(Alphabet_Mapping_List[(int)(digit_Name)])

                count = 0

                print ("")

                print ("Prcessing Alphabet - " + str (last_digit_Name))



            image_Path = image_Folder_Path + '/' + last_digit_Name + '/' + str(last_digit_Name) + '-' + str(count) + '.png'

            new_image.save(image_Path)

            count = count + 1



            if count % 1000 == 0:

                print ("Images processed: " + str(count))
convert_csv_to_image(training_csv,image_Folder_Path_training )
convert_csv_to_image(test_csv, image_Folder_Path_test)
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout



#creating a 2-layer CNN



classifier = Sequential()



classifier.add(Conv2D(64, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Dropout(0.2))



classifier.add(Conv2D(128, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Dropout(0.2))



classifier.add(Flatten())



classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 26, activation='softmax'))



classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# In this section using ImageDataGenerator we will create real time data augmentation

from keras.preprocessing.image import ImageDataGenerator



#augmenting training data 

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



#reading the images from training folder

training_set = train_datagen.flow_from_directory('/kaggle/image/training-set',

                                                 target_size = (28, 28),

                                                 batch_size = 32,

                                                 class_mode = 'categorical',color_mode ="grayscale")
# reading the images from testing folder

test_set = test_datagen.flow_from_directory('/kaggle/image/test-set',

                                                 target_size = (28, 28),

                                                 batch_size = 32,

                                                 class_mode = 'categorical',color_mode ="grayscale")
# creating different callback options

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



mcp = ModelCheckpoint(filepath='cnn_model.h5',monitor='val_loss',verbose=1,save_best_only=True)

es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
#fitting the model

classifier.fit_generator(training_set,

                         steps_per_epoch = len(training_set),

                         epochs = 5,

                         validation_data = test_set,

                         validation_steps = len(test_set),

                         callbacks =[es,rlr,mcp])
import matplotlib.pyplot as plt

plt.plot(classifier.history.history['acc'])

plt.plot(classifier.history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(classifier.history.history['loss'])

plt.plot(classifier.history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()