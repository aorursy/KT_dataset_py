import os

from PIL import Image

import numpy as np

from IPython.display import display

import h5py 
# Path to your dataset folder..........

path = '../input/101_ObjectCategories'

os.chdir(path)
folders = os.listdir()

# print(folders)
img_size = 256
#making image square for efficient training....

#For reference: https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black/44231784

def make_square(image, min_size=img_size, fill_color=(0, 0, 0, 0)):

    size = (min_size, min_size)

    image.thumbnail(size, Image.ANTIALIAS)

    background = Image.new('RGB', size, (255, 255, 255, 0))

    background.paste(

        image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2))

    )

    new_img = np.array(background, dtype=np.uint8)/255

    return new_img
#checking number of images in dataset......

noOfImages = 0

for folder in range(len(folders)):

  pathFolder = str(folders[folder]) + "/"

  os.chdir(pathFolder)

  folderImages = os.listdir()



  for image in range(len(folderImages)):

    noOfImages += 1

    







  os.chdir("..")

  

print("Number of Images",noOfImages)

  
#Processing each image and converting to array....................

datasetImages = np.zeros((noOfImages,img_size,img_size,3)) #allocating memory for efficient storage, avoid using python list and then appending..............

datasetClasses = []

count = 0



for folder in range(len(folders)):

  pathFolder = str(folders[folder]) + "/"

  os.chdir(pathFolder)

  folderImages = os.listdir()

  

  for image in range(len(folderImages)):

    img = Image.open(folderImages[image])

    imageArray = make_square(img)

    datasetImages[count] = imageArray

    datasetClasses.append(folder)

    count += 1

    

  os.chdir("..")



datasetClasses = np.array(datasetClasses)
import matplotlib.pyplot as plt

plt.imshow(datasetImages[0])

# Changing directory to working directory in case of Kaggle, For others leave as it is.........

os.chdir("..")

os.chdir("..")

os.chdir("working")
# Checking current directory..............

print(os.getcwd())
print(type(datasetImages))

print(type(datasetClasses))
# Storing image array into HDF5 structure for faster and efficient storage.....

#For reference: https://support.hdfgroup.org/HDF5/doc/H5.intro.html

#             : https://realpython.com/storing-images-in-python/

def store_into_hdf5(imagesArray,labelsArray):

  num_images = len(imagesArray)

  file = h5py.File("imageDataset.h5","w")

      # Create a dataset in the file

  dataset = file.create_dataset(

      "images", (len(imagesArray),img_size,img_size,3), h5py.h5t.STD_U8BE, data=imagesArray

  )

  meta_set = file.create_dataset(

      "meta", (len(labelsArray),1) , h5py.h5t.STD_U8BE, data=labelsArray

  )

  file.close()

store_into_hdf5( datasetImages, datasetClasses )
# free RAM memory when variable is used.........................

del datasetImages
del datasetClasses
# incase you want to remove file from Kaggle

# os.remove("imageDataset.h5")
# Reading image array but in compressed mode, you can see image size has been reduced..............

def read_many_hdf5():

    """ Reads image from HDF5.

        Parameters:

        ---------------

        num_images   number of images to read



        Returns:

        ----------

        images      images array, (N, 32, 32, 3) to be stored

        labels      associated meta data, int label (N, 1)

    """

    images, labels = [], []



    # Open the HDF5 file

    file = h5py.File("imageDataset.h5", "r+")



    images = np.array(file["/images"]).astype("uint8")

    labels = np.array(file["/meta"]).astype("uint8")



    return images, labels
datasetImages, datasetClasses = read_many_hdf5()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    datasetImages, datasetClasses, test_size=0.3, random_state=42)
import keras

num_classes = len(np.unique(datasetClasses))
# convert class vectors to binary class matrices

# For reference: https://stackoverflow.com/a/53430549

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
from __future__ import print_function

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten , BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

batch_size = 128

epochs = 12
#For using TPU, incase using gpu don't run this cell...............

import tensorflow as tf

# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope(): # Remove this line incase you are not using TPU..............

    model = Sequential()

    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    #model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation = 'softmax')) #As number of nodes in last layer in softmax is number of classes, where each node is probability of classes

    

    model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])
# Data Augmentation for deep network to identify images correctly..

# For Reference: https://www.quora.com/What-is-data-augmentation-in-deep-learning

from keras.preprocessing.image import ImageDataGenerator

# performing data argumentation by training image generator

dataAugmentaion = ImageDataGenerator(rotation_range = 30, zoom_range = 0.20, 

fill_mode = "nearest", shear_range = 0.20, horizontal_flip = True, 

width_shift_range = 0.1, height_shift_range = 0.1)
#Training in batches, you can use fit() for smaller dataset training.......

#For Refernce : https://datascience.stackexchange.com/a/34452

model.fit_generator(dataAugmentaion.flow(X_train, y_train, batch_size = 32),

 validation_data = (X_test, y_test), steps_per_epoch = len(X_train) // 32,

 epochs = 10)
# Calculating performance of the model................

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])