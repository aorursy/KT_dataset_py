# Cancer Detection project, WIP - Rory Gold

# Aim is to gain familiarity with the use of CNNs with tensorflow for image classification



# CNN structure is based off advice from online Data Science posts and Kaggle Deep Learning microcourse
# Import libraries

import os



import numpy as np

import pandas as pd

import random

import math 

import time

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV



from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D

from sklearn.model_selection import train_test_split

from tensorflow.python import keras



from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping



from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.python.keras.constraints import maxnorm

from tensorflow.keras.optimizers import Adam



from os.path import join

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.applications.resnet50 import preprocess_input

# print('Stage complete')
# Define Functions

def get_filepaths(dirname):

    img_paths = []

    for _,_,filenames in os.walk(dirname):

        for filename in filenames:

            img_paths.append(os.path.join(dirname, filename))

    return img_paths, filenames



def get_file_ids(filenames):

    file_ids = []

    splt_fstop = '.'

    for filename in filenames:

        file_ids.append(filename.partition(splt_fstop)[0])

    return file_ids



def pandas_to_array(data):

    data_array = data.to_numpy()

    return data_array



def OHE_encode_target(target):

    OHE_target = pd.get_dummies(target)

    OHE_target.columns = ['No_Cancer', 'Cancer']

    return OHE_target



def balance_sample_classes(train_labels, img_paths, img_range):

    train_labels = train_labels.iloc[img_range]

    number_samples = len(train_labels)

    number_cancer = sum(train_labels.label)

    class_balance = number_cancer/number_samples

    half_samples = number_samples/2



    noncancer_index = train_labels.index[train_labels.label==0]

    cancer_index = train_labels.index[train_labels.label==1]



    balance_difference = abs(number_cancer-half_samples)

    # If less cancer labels, need to reduce non cancer labels, drop difference number of non cancer

    if number_cancer < half_samples:

        noncancer_index = noncancer_index#[0:(len(noncancer_index)-int(2*balance_difference))]

    if number_cancer > half_samples:

        cancer_index = cancer_index#[0:(len(cancer_index)-int(2*balance_difference))]

    

    # Above code commented out as experimenting with 0 class balancing in the hope that randomized dataset will give representation

    # of spread of data

        

    balanced_index = np.concatenate([noncancer_index, cancer_index])

    balanced_index_sorted = np.sort(balanced_index)



    train_labels_balanced = train_labels.loc[balanced_index_sorted]

    train_labels = train_labels_balanced.reset_index(drop=True)



    img_paths_balanced = []

    for index in balanced_index_sorted:

        img_paths_balanced.append(img_paths[index])

    img_paths = img_paths_balanced

    return train_labels, img_paths



image_size=96

def read_and_prep_images_ver3(img_paths, img_height=image_size, img_width=image_size):

    img_arrays = []

    for img_path in img_paths:

        img = load_img(img_path, target_size=(img_height, img_width))

        img_arrays.append(img_to_array(img))

        img.close

    img_array = np.array(img_arrays)

    

    # Scale images between 0 and 1

    output = img_array/255

    return output



def make_range_listing(list_range):

    index_list = []

    for index in list_range:

        index_list.append(index)

    return index_list



def shuffle_listing(listing):

    shuffled_listing = []

    for index in index_list:

        shuffled_listing.append(listing[index])

    return shuffled_listing



# Create function to generate train batch data for a specific image range

def generate_train_data(img_range, img_paths_train, train_file_ids): 

    # Insert labels for each file

    train_file = '/kaggle/input/histopathologic-cancer-detection/train_labels.csv'

    train_labels = pd.read_csv(train_file)



    # Sort train_labels by train image listing order

    train_labels['id_cat'] = pd.Categorical(train_labels.id, categories = train_file_ids, ordered= True)

    train_labels = train_labels.sort_values('id_cat')

    train_labels = train_labels.reset_index(drop=True)

    del train_labels['id_cat']



    train_labels, img_paths_train = balance_sample_classes(train_labels, img_paths_train, img_range)

    train_data = read_and_prep_images_ver3(img_paths_train)



    train_Y = OHE_encode_target(train_labels.label)

    train_Y = pandas_to_array(train_Y)

    return train_data, train_Y



#print('Stage complete')
# Set random seed

np.random.seed(1)
# Choose images to work with

dirname_train = '/kaggle/input/histopathologic-cancer-detection/train/'

dirname_test = '/kaggle/input/histopathologic-cancer-detection/test/'



img_paths_train, filenames_train = get_filepaths(dirname_train)

train_file_ids = get_file_ids(filenames_train)

#print('Stage complete')
# Insert code to randomize listing - simply randomized listing entries

list_range = range(0,len(img_paths_train))

index_list = make_range_listing(list_range)

random.shuffle(index_list)



# Shuffle path listings

img_paths_train = shuffle_listing(img_paths_train)

filenames_train = shuffle_listing(filenames_train)

train_file_ids = shuffle_listing(train_file_ids)
def Create_Model(image_size, num_classes):

    img_width = image_size

    img_height = image_size

    model = Sequential()

    model.add(Conv2D(200, kernel_size=(3,3), strides=2, activation='relu', input_shape=(img_height, img_width, 3)))

    model.add(AveragePooling2D())

    model.add(Dropout(0.3))

    model.add(Conv2D(200, kernel_size=(3,3), strides=2, activation='relu'))

    model.add(AveragePooling2D())

    model.add(Dropout(0.3))

    model.add(Conv2D(200, kernel_size=(3,3), strides=2, activation='relu'))

    model.add(AveragePooling2D())

    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    return model



def train_model(model, train_data, train_Y, callbacks_list):

    model.fit(train_data, train_Y, batch_size=128, epochs=150, validation_split=0.1, callbacks = callbacks_list)

    return model
# Create model

image_size = 96

num_classes = 2



# Define early stopping parameters

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, patience=10)

callbacks_list = [checkpoint, es]

model = Create_Model(image_size, num_classes)



# Compile model

model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
# Fit model with early stopping

# Define img_range

img_range = range(0,35000)



# Generate training data

train_data, train_Y = generate_train_data(img_range, img_paths_train, train_file_ids)

#print('Stage Complete')

model = train_model(model, train_data, train_Y, callbacks_list)
# Clear training data to clear RAM for next stage

del train_data, train_Y
# Generate validation data

img_range_val = range(210000, 220025)

validation_data, validation_Y = generate_train_data(img_range_val, img_paths_train, train_file_ids)



# Evaluate model so far

model.load_weights("weights.best.hdf5")

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

scores = model.evaluate(validation_data, validation_Y)



# Delete validation data and show scores

del validation_data, validation_Y

print('My model scored', scores)