import h5py

import os
class HDF5DatasetWriter:

    def __init__(self, dims, outputPath, dataKey="images",bufSize=1000):

    # check to see if the output path exists, and if so, raise

    # an exception

        if os.path.exists(outputPath):

            raise ValueError("The supplied ‘outputPath‘ already "

         "exists and cannot be overwritten. Manually delete "

         "the file before continuing.", outputPath)



         # open the HDF5 database for writing and create two datasets:

         # one to store the images/features and another to store the

         # class labels

        self.db = h5py.File(outputPath, "w")

        self.data = self.db.create_dataset(dataKey, dims,dtype="float")

        self.labels = self.db.create_dataset("labels", (dims[0],),dtype="int")



         # store the buffer size, then initialize the buffer itself

         # along with the ind

        self.bufSize = bufSize

        self.buffer = {"data": [], "labels": []}

        self.idx = 0

    def add(self, rows, labels):

        # add the rows and labels to the buffer

        self.buffer["data"].extend(rows)

        self.buffer["labels"].extend(labels)

        

        # check to see if the buffer needs to be flushed to disk

        if len(self.buffer["data"]) >= self.bufSize:

            self.flush()

    def flush(self):

        # write the buffers to disk then reset the buffer

        i = self.idx + len(self.buffer["data"])

        self.data[self.idx:i] = self.buffer["data"]

        self.labels[self.idx:i] = self.buffer["labels"]

        self.idx = i

        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):

        # create a dataset to store the actual class label names,

        # then store the class labels

        dt = h5py.special_dtype()

        labelSet = self.db.create_dataset("label_names",

        (len(classLabels),), dtype=dt)

        labelSet[:] = classLabels

    def close(self):

         # check to see if there are any other entries in the buffer

         # that need to be flushed to disk

        if len(self.buffer["data"]) > 0:

            self.flush()

        

         # close the dataset

        self.db.close()
# Config file 



## Dataset paths

animal_dataset_path = '../input/animal-image-datasetdog-cat-and-panda/'

flower_dataset_path = '../input/flowers17/17flowers/jpg'

caltech101_path = '../input/caltech-101/caltech101'



## output Paths to save the extracted features file i.e. '.hdf5 file'



animal_hdf5_path = 'animal_features.hdf5'

flowers_hdf5_path= 'flowers_features.hdf5'

caltech_hdf5_path = 'caltech_features.hdf5'





#Batch size 



batch_size = 32



buffer = 1000



! pip install imutils

! pip install  progressbar
import numpy as np

import random

from imutils import paths

from tensorflow.keras.applications import VGG16

from tensorflow.keras.applications import imagenet_utils

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import load_img

from sklearn.preprocessing import LabelEncoder

import progressbar

import os



print("[INFO] loading images...")

imagePaths = list(paths.list_images(animal_dataset_path))

random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the

# labels

labels = [p.split(os.path.sep)[-2] for p in imagePaths]

le = LabelEncoder()

labels = le.fit_transform(labels)
print("[INFO] loading network...")

model = VGG16(weights="imagenet", include_top=False)
# initialize the HDF5 dataset writer, then store the class label

# names in the dataset

dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7),animal_hdf5_path, dataKey="features")





widgets = ["Extracting Features: ", progressbar.Percentage(), " ",

progressbar.Bar(), " ", progressbar.ETA()]

pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()



# loop over the images in patches

for i in np.arange(0, len(imagePaths), batch_size):

# extract the batch of images and labels, then initialize the

# list of actual images that will be passed through the network

# for feature extraction

    batchPaths = imagePaths[i:i + batch_size]

    batchLabels = labels[i:i + batch_size]

    batchImages = []

    # loop over the images and labels in the current batch

    for (j, imagePath) in enumerate(batchPaths):

      # load the input image using the Keras helper utility

      # while ensuring the image is resized to 224x224 pixels

        image = load_img(imagePath, target_size=(224, 224))

        image = img_to_array(image)



        # preprocess the image by (1) expanding the dimensions and

        # (2) subtracting the mean RGB pixel intensity from the

        # ImageNet dataset

        image = np.expand_dims(image, axis=0)

        image = imagenet_utils.preprocess_input(image)



        # add the image to the batch

        batchImages.append(image)

    batchImages = np.vstack(batchImages)

    features = model.predict(batchImages, batch_size=batch_size)

    # reshape the features so that each image is represented by

    # a flattened feature vector of the ‘MaxPooling2D‘ outputs

    features = features.reshape((features.shape[0], 512 * 7 * 7))

    

    # add the features and labels to our HDF5 dataset

    dataset.add(features, batchLabels)

    pbar.update(i)

# close the dataset

dataset.close()

pbar.finish()



print("[INFO] loading images...")

imagePaths = list(paths.list_images(flower_dataset_path))

random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the

# labels

labels = [p.split(os.path.sep)[-2] for p in imagePaths]

le = LabelEncoder()

labels = le.fit_transform(labels)
# initialize the HDF5 dataset writer, then store the class label

# names in the dataset

dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7),flowers_hdf5_path, dataKey="features")

#dataset.storeClassLabels(le.classes_)



widgets = ["Extracting Features: ", progressbar.Percentage(), " ",

progressbar.Bar(), " ", progressbar.ETA()]

pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()



# loop over the images in patches

for i in np.arange(0, len(imagePaths), batch_size):

# extract the batch of images and labels, then initialize the

# list of actual images that will be passed through the network

# for feature extraction

    batchPaths = imagePaths[i:i + batch_size]

    batchLabels = labels[i:i + batch_size]

    batchImages = []

    # loop over the images and labels in the current batch

    for (j, imagePath) in enumerate(batchPaths):

      # load the input image using the Keras helper utility

      # while ensuring the image is resized to 224x224 pixels

        image = load_img(imagePath, target_size=(224, 224))

        image = img_to_array(image)



        # preprocess the image by (1) expanding the dimensions and

        # (2) subtracting the mean RGB pixel intensity from the

        # ImageNet dataset

        image = np.expand_dims(image, axis=0)

        image = imagenet_utils.preprocess_input(image)



        # add the image to the batch

        batchImages.append(image)

    batchImages = np.vstack(batchImages)

    features = model.predict(batchImages, batch_size=batch_size)

    # reshape the features so that each image is represented by

    # a flattened feature vector of the ‘MaxPooling2D‘ outputs

    features = features.reshape((features.shape[0], 512 * 7 * 7))

    

    # add the features and labels to our HDF5 dataset

    dataset.add(features, batchLabels)

    pbar.update(i)

# close the dataset

dataset.close()

pbar.finish()
print("[INFO] loading images...")

imagePaths = list(paths.list_images(caltech101_path))

random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the

# labels

labels = [p.split(os.path.sep)[-2] for p in imagePaths]

le = LabelEncoder()

labels = le.fit_transform(labels)
# initialize the HDF5 dataset writer, then store the class label

# names in the dataset

dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7),caltech_hdf5_path, dataKey="features")

#dataset.storeClassLabels(le.classes_)



widgets = ["Extracting Features: ", progressbar.Percentage(), " ",

progressbar.Bar(), " ", progressbar.ETA()]

pbar = progressbar.ProgressBar(maxval=len(imagePaths),widgets=widgets).start()



# loop over the images in patches

for i in np.arange(0, len(imagePaths), batch_size):

# extract the batch of images and labels, then initialize the

# list of actual images that will be passed through the network

# for feature extraction

    batchPaths = imagePaths[i:i + batch_size]

    batchLabels = labels[i:i + batch_size]

    batchImages = []

    # loop over the images and labels in the current batch

    for (j, imagePath) in enumerate(batchPaths):

      # load the input image using the Keras helper utility

      # while ensuring the image is resized to 224x224 pixels

        image = load_img(imagePath, target_size=(224, 224))

        image = img_to_array(image)



        # preprocess the image by (1) expanding the dimensions and

        # (2) subtracting the mean RGB pixel intensity from the

        # ImageNet dataset

        image = np.expand_dims(image, axis=0)

        image = imagenet_utils.preprocess_input(image)



        # add the image to the batch

        batchImages.append(image)

    batchImages = np.vstack(batchImages)

    features = model.predict(batchImages, batch_size=batch_size)

    # reshape the features so that each image is represented by

    # a flattened feature vector of the ‘MaxPooling2D‘ outputs

    features = features.reshape((features.shape[0], 512 * 7 * 7))

    

    # add the features and labels to our HDF5 dataset

    dataset.add(features, batchLabels)

    pbar.update(i)

# close the dataset

dataset.close()

pbar.finish()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

import argparse

import pickle

import h5py
animal_hdf5_path =  'animal_features.hdf5'

flowers_hdf5_path = 'flowers_features.hdf5'

caltech101_hdf5_path ='caltech_features.hdf5'



model_animal_path = 'animal_model01.cpickel'



model_flowers_path = 'flowers_model01.cpickel'



model_caltech_path = 'caltech_model01.cpickel'

 
# open the HDF5 database for reading then determine the index of

# the training and testing split, provided that this data was

# already shuffled *prior* to writing it to disk

db = h5py.File(animal_hdf5_path, "r")

i = int(db["labels"].shape[0] * 0.75)
# define the set of parameters that we want to tune then start a

# grid search where we evaluate our model for each value of C

print("[INFO] tuning hyperparameters...")

params = {"C": [0.1, 1.0, 10.0]}

model = GridSearchCV(LogisticRegression(), params, cv=3,

n_jobs=-1)

model.fit(db["features"][:i], db["labels"][:i])

print("[INFO] best hyperparameters: {}".format(model.best_params_))



# evaluate the model

print("[INFO] evaluating...")

preds = model.predict(db["features"][i:])

print(classification_report(db["labels"][i:], preds))
# serialize the model to disk

print("[INFO] saving model...")

f = open(model_animal_path, "wb")

f.write(pickle.dumps(model.best_estimator_))

f.close()



# close the database

db.close()
# open the HDF5 database for reading then determine the index of

# the training and testing split, provided that this data was

# already shuffled *prior* to writing it to disk

db = h5py.File(flowers_hdf5_path, "r")

i = int(db["labels"].shape[0] * 0.75)
# define the set of parameters that we want to tune then start a

# grid search where we evaluate our model for each value of C

print("[INFO] tuning hyperparameters...")

params = {"C": [0.1, 1.0]}

model = GridSearchCV(LogisticRegression(), params, cv=3,

n_jobs=-1)

model.fit(db["features"][:i], db["labels"][:i])

print("[INFO] best hyperparameters: {}".format(model.best_params_))



# evaluate the model

print("[INFO] evaluating...")

preds = model.predict(db["features"][i:])

print(classification_report(db["labels"][i:], preds))
# serialize the model to disk

print("[INFO] saving model...")

f = open(model_flowers_path, "wb")

f.write(pickle.dumps(model.best_estimator_))

f.close()



# close the database

db.close()
# open the HDF5 database for reading then determine the index of

# the training and testing split, provided that this data was

# already shuffled *prior* to writing it to disk

db = h5py.File(caltech101_hdf5_path, "r")

i = int(db["labels"].shape[0] * 0.75)
# define the set of parameters that we want to tune then start a

# grid search where we evaluate our model for each value of C

# print("[INFO] tuning hyperparameters...")

# params = {"C": [1.0]}

# model = GridSearchCV(LogisticRegression(), params, cv=3,

# n_jobs=-1)

# model.fit(db["features"][:i], db["labels"][:i])

# print("[INFO] best hyperparameters: {}".format(model.best_params_))



model = LogisticRegression(C=0.1)

model.fit(db["features"][:i], db["labels"][:i])





# evaluate the model

print("[INFO] evaluating...")

preds = model.predict(db["features"][i:])

print(classification_report(db["labels"][i:], preds))
# serialize the model to disk

print("[INFO] saving model...")

f = open(model_caltech_path, "wb")

f.write(pickle.dumps(model))

f.close()



# close the database

db.close()