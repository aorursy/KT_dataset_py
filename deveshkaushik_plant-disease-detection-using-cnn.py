import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np

import PIL

from PIL import Image

from skimage.color import rgb2gray

from scipy import ndimage as ndi

import cv2

import os

from os import listdir

from sklearn.utils import shuffle



import keras

from keras.models import Sequential, load_model

from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from keras.optimizers import SGD

from keras import regularizers

from keras.callbacks import ModelCheckpoint

directory_root = "../input/plantvillage/PlantVillage"

print(len(listdir(directory_root)))
image_list, label_list = [], []

try:

    print("[INFO] Loading images ...")

    root_dir = listdir(directory_root)

    for directory in root_dir :

        # remove .DS_Store from list

        if directory == ".DS_Store" :

            root_dir.remove(directory)



    for plant_folder in root_dir :

        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")

                

        for single_plant_disease_image in plant_disease_folder_list :

            if single_plant_disease_image == ".DS_Store" :

                plant_disease_folder_list.remove(single_plant_disease_image)



        for image in plant_disease_folder_list:

            image_directory = f"{directory_root}/{plant_folder}/{image}"

            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:

                image_list.append(image_directory)

                label_list.append(plant_folder)

    print("[INFO] Image loading completed")  

except Exception as e:

    print(f"Error : {e}")
img_info = pd.DataFrame({'image_path':image_list,'label':label_list})

print(img_info.head())

print(len(img_info))
#new column (empty)

img_info["labels_integer"] = None

#index of new column

index_labels_integer = img_info.columns.get_loc("labels_integer")

#index of species column

index_species = img_info.columns.get_loc("label")

#to assign numeric labels starting with 0 for the first species

k = 0 

for i in range(len(img_info)):

    if i == 0:

        img_info.iloc[i, index_labels_integer] = k #here, k == 0

    if i > 0:

        if img_info.iloc[i-1, index_species] == img_info.iloc[i, index_species]:

            img_info.iloc[i, index_labels_integer] = k

        else:

            k += 1

            img_info.iloc[i, index_labels_integer] = k

img_info.tail()
img_info = shuffle(img_info)

list_vectors = []



for image_path in img_info.image_path:

    #read as rgb array

    img = Image.open(image_path)

    size = (64, 64)

    img = img.resize(size, PIL.Image.ANTIALIAS)  

    img_array = np.array(img)

    #append image vector to list

    list_vectors.append(img_array)

    

print(len(list_vectors))
X = np.stack((list_vectors))

Y =  img_info['labels_integer']

print(X.shape)
X = X/255

Y_one_hot = keras.utils.to_categorical(Y, num_classes=15)

print(Y.shape, Y_one_hot.shape)
np.savez("x_images_arrayscnn", X)

np.savez("y_numeric_labelscnn", Y_one_hot)
x_npz = np.load("x_images_arrayscnn.npz")

X = x_npz['arr_0']



y_npz = np.load("y_numeric_labelscnn.npz")

Y_one_hot = y_npz['arr_0']



print(X.shape)
split_train = 0.8 #train 0.8, validate 0.1, test 0.1

split_val = 0.9

index_train = int(split_train*len(X))

index_val = int(split_val*len(X))



X_train = X[:index_train]

X_val = X[index_train:index_val]

X_test = X[index_val:]



Y_train = Y_one_hot[:index_train]

Y_val = Y_one_hot[index_train:index_val]

Y_test = Y_one_hot[index_val:]





print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) 

num_classes = 15



model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(64, (5, 5)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.7))

model.add(Flatten())

model.add(Dense(1000))

model.add(Activation('relu'))

model.add(Dropout(0.7))

model.add(Dense(num_classes, activation='softmax'))





model.compile(optimizer=keras.optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
best_model_file = "disease_selected_100.h5"

best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)



print('Training model...')

results = model.fit(X_train, Y_train, epochs=200, batch_size=64, validation_data=(X_val, Y_val), callbacks=[best_model])

print('Traing finished.')



print('Loading the best model...')

model = load_model(best_model_file)

print('Best Model loaded!')
model.evaluate(X_test, Y_test)