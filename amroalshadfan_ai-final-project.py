# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# linear algebra & to deal with parallel matrecies processing and operations
import numpy as np 

# data processing, CSV file I/O
import pandas as pd 

# generate random integer numbers
from random import randint 

# library to plot graphs
import matplotlib.pyplot as plt
%matplotlib inline

# tensorflow library for image processing
import tensorflow as tf

# deal with files paths using different notations (*, _, %)
import glob

# deal with paths and directories 
import os

# OpenCV Which deals with image processing
import cv2

# Keras library for creating the CNN model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import LSTM, Input, TimeDistributed
from keras.models import Model
from keras.callbacks.callbacks import History
from keras.optimizers import RMSprop, SGD
from keras import backend as K

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# Print all folders available in the input directory.
print("Folder in the input directory (Dataset name):", os.listdir("../input"))

# Print all folders available in the input directory.
print("Folders in gemstones-images directory:", os.listdir("../input/gemstones-images")) 

# Print the number of folder in testing & training folders
print("\n-----Directories & Files Count-----")
print("Testing Folders Count:", len(os.listdir("../input/gemstones-images/test")), "\nTraining Folders Count:", len(os.listdir("../input/gemstones-images/train")))

# variables to count testing & training images
testingImagesCount = 0 
trainingImagesCount = 0

# loop on all files in each folder
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        if dirname.split('/')[-2] == "test":
            testingImagesCount += 1
        else:
            trainingImagesCount += 1
        # print full directory path
        #print("Directory:",os.path.join(dirname, filename), "\n# of files:", len(os.listdir(dirname)))
        
print("# of Testing Images:", testingImagesCount)  
print("# of Training Images:", trainingImagesCount)
# Function to get images and their lables and add them to their corresponding variables
def extractImagesWithLabeling(image_folder, images_set, label_set):
    # loop on all images in the folder path
    image_label = image_folder.split('/')[-1]
    for image_path in glob.glob(os.path.join(image_folder, '*')):
        # Obtain the last element in the array ["..", "input", "gemstones-images", "test", ("Carnelian")]
        label_set.append(image_label) 
        
        # read the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) # (imagePath, imageColorMode)
        
        # resize the image to reduce number of parameters
        image = cv2.resize(image, (45, 45)) # Change image dimensions from 100x100 to 45x45 (to reduce number of params)
        
        # process the image & add it to the referenced variable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Change image color mode from RGB to BGR, since openCV deals with BGR.
        images_set.append(image)

training_gems_images_set = [] # Matix where images appended (training set)
training_gems_labels_set = [] # Lables tagged to images (training set)

testing_gems_images_set = [] # Matix where images appended (testing set)
testing_gems_labels_set = [] # Lables tagged to images (testing set)

# sets variable contains the reference of the variables above so they can be send correspondingly to extractImagesWithLabeling
sets = [[training_gems_images_set, training_gems_labels_set], [testing_gems_images_set, testing_gems_labels_set]]

for index, path in enumerate(glob.glob("../input/gemstones-images/*")):
    #path_split = path.split('/')[-1]
    for image_folder in glob.glob(os.path.join(path,'*')):
        #print(image_folder)
        # extractImagesWithLabeling(image_folder, referencing images_sets vairables, referencing labels_sets vairables)
        extractImagesWithLabeling(image_folder, sets[index][0], sets[index][1]) 

print("-----Numbers & Statistics-----")
print("# of Images in the training set:", len(training_gems_images_set))
print("# of Labels in the training set:", len(training_gems_labels_set))

print("# of Images in the testing set:", len(testing_gems_images_set))
print("# of Labels in the testing set:", len(testing_gems_labels_set))

# Transform variables array to numpy array for parallel compuation to increase performance
training_gems_images_set = np.array(training_gems_images_set)
training_gems_labels_set = np.array(training_gems_labels_set)

testing_gems_images_set = np.array(testing_gems_images_set)
testing_gems_labels_set = np.array(testing_gems_labels_set)

# Printing results for debugging purposes
# print('training_gems_images_set Array:', training_gems_images_set, '\n')
# print('training_gems_labels_set Labels:', training_gems_labels_set, '\n')

# print('testing_gems_images_set Array:', testing_gems_images_set, '\n')
# print('testing_gems_labels_set Labels:', testing_gems_labels_set, '\n')

# Save a copy of the testing lables for future testing
actual_results_for_testing = testing_gems_labels_set

#print(testing_gems_labels_set)
# Create a dictionary to index the lables
label_to_id_dict_train_set = {}
for index, label in enumerate(np.unique(training_gems_labels_set)):
    label_to_id_dict_train_set[label] = index

# Create a dictionary lables to index
id_to_label_dict_train_set = {}
for label, index in label_to_id_dict_train_set.items():
    id_to_label_dict_train_set[index] = label
    
print('Labels:Index - Dictionary Format:', label_to_id_dict_train_set, '\n') 
print('Index:Labels - Dictionary Format:', id_to_label_dict_train_set, '\n') 

label_ids_train_set = np.array([label_to_id_dict_train_set[label] for label in training_gems_labels_set])
print(label_ids_train_set)

print("training_gems_images_set Shape:", training_gems_images_set.shape, '\nlabel_ids_train_set Shape:', label_ids_train_set.shape, '\ntraining_gems_labels_set Shape:', training_gems_labels_set.shape) 
# Create a dictionary to index the lables
label_to_id_dict_test_set = {}
for index, label in enumerate(np.unique(testing_gems_labels_set)):
    label_to_id_dict_test_set[label] = index

# Create a dictionary lables to index
id_to_label_dict_test_set = {}
for label, index in label_to_id_dict_test_set.items():
    id_to_label_dict_test_set[index] = label

print('Labels:Index - Dictionary Format:', label_to_id_dict_test_set, '\n') 
print('Index:Labels - Dictionary Format:', id_to_label_dict_test_set, '\n') 

label_ids_test_set = np.array([label_to_id_dict_test_set[label] for label in testing_gems_labels_set])
print(label_ids_test_set)

print("testing_gems_images_set Shape:", testing_gems_images_set.shape, '\nlabel_ids_test_set Shape:', label_ids_test_set.shape, '\ntesting_gems_labels_set Shape:', testing_gems_labels_set.shape)
# Normalize the images between 0-1
training_gems_images_set = training_gems_images_set/255
testing_gems_images_set = testing_gems_images_set/255

# Print the images matrices
#print(training_gems_images_set)

# Connect id with lable and convert lables to readable categorical form for the model
training_gems_labels_set = keras.utils.to_categorical(label_ids_train_set, 87)
testing_gems_labels_set = keras.utils.to_categorical(label_ids_test_set, 87)

# Falttening the data to have an idea how it will look
training_gems_images_flat_set = training_gems_images_set.reshape(training_gems_images_set.shape[0], 45*45*3)
testing_gems_images_flat_set = testing_gems_images_set.reshape(testing_gems_images_set.shape[0], 45*45*3)

# print(training_gems_images_flat_set.shape)
# print(testing_gems_images_flat_set.shape)
print('Original Sizes:', training_gems_images_set.shape, training_gems_labels_set.shape, testing_gems_images_set.shape, testing_gems_labels_set.shape)
print('Flattened:', training_gems_images_flat_set.shape, testing_gems_images_flat_set.shape)
# Print image
plt.imshow(training_gems_images_set[randint(0, len(training_gems_images_set)-1)])
plt.title("Image of a random stone")
plt.show()
# Function to create the model instance
def create_gems_cnn_model():

    # create sequential model where layers are linear
    model_cnn = Sequential()

    # first convolutional layer
    # Conv2D(number of filters, filter shape, activation function, image shape)
    model_cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(45, 45, 3))) # Relu activation function results are the same value if above 0 and 0 if the value is < 0

    # Conv2D(number of filters, filter shape, activation function)
    model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    model_cnn.add(Conv2D(64, (3, 3), activation='relu'))


    # Max pooling to reduce the # of parameters
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    model_cnn.add(Conv2D(128, (3, 3), activation='relu'))
    model_cnn.add(Conv2D(128, (3, 3), activation='relu'))


    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    model_cnn.add(Conv2D(256, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    model_cnn.add(Dropout(0.25)) # number of synaptic weights to leave and not to update
    model_cnn.add(Flatten()) # transform image inputs into a 1D array for input to the fully connected Neural Network

    model_cnn.add(Dense(128, activation='relu'))

    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(87, activation='softmax')) # activation function Softmax output is converted to a sum of 1

    model_cnn.compile(loss=keras.losses.categorical_crossentropy, # Loss is depending on category not on value such as 2 != 1
              # loss = summation(ydesired*log(yactual))
              optimizer="Adamax",
              metrics=['accuracy'])

    return model_cnn
gems_model = create_gems_cnn_model()
gems_model.fit(training_gems_images_set, training_gems_labels_set, batch_size=70, epochs=121,verbose=0)

score = gems_model.evaluate(testing_gems_images_set, testing_gems_labels_set, verbose=1)
# Plotting the process of training
plt.plot(gems_model.history.history['loss'])
plt.plot(gems_model.history.history['accuracy'])
plt.title('Accuracy Timeline')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['Loss', 'Accuray'], loc='upper left')
plt.show()
# Evaluating the model on testing data & printing accuracy
model_evaluation = gems_model.evaluate(testing_gems_images_set, testing_gems_labels_set,verbose=0)
print('Test Accuracy:', model_evaluation[1]*100)
# Predict the testing images using our model
model_testing = gems_model.predict_classes(testing_gems_images_set)

# Display the images and print the real vs the predicted results
for index, itemResult in enumerate(model_testing):
    plt.imshow(testing_gems_images_set[index])
    plt.title("Predication Result: " + id_to_label_dict_test_set[itemResult] + "\nReal Result: " + actual_results_for_testing[index])
    plt.show()