!ls ../input/data/data
import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

import random



import cv2

from tqdm import tqdm

from random import shuffle

from zipfile import ZipFile

from PIL import Image



from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization



from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD

from keras import backend as K



from keras import applications

from keras.layers import Dense

from keras.models import Model



from keras.models import model_from_json

# 5 folders for 5 flower types: daisy, dandelion, rose, sunflower, tulip

path_daisy = '../input/data/data/daisy'

path_dandelion = '../input/data/data/dandelion'

path_rose = '../input/data/data/rose'

path_sunflower = '../input/data/data/sunflower'

path_tulip = '../input/data/data/tulip'
# Get size of each class

size_daisy = len(os.listdir(path_daisy))

size_dandelion = len(os.listdir(path_dandelion))

size_rose = len(os.listdir(path_rose))

size_sunflower = len(os.listdir(path_sunflower))

size_tulip = len(os.listdir(path_tulip))



# Print size of each class

print('daisy data size: {}'.format(size_daisy))

print('dandelion data size: {}'.format(size_dandelion))

print('rose data size: {}'.format(size_rose))

print('sunflower data size: {}'.format(size_sunflower))

print('tulip data size: {}'.format(size_tulip))

# Display images

def display_images(path, nbr_images_to_display, rows, columns):

    """

    Takes as input the path, number of images to display, number of rows and columns and display random

    images images organized along rows and columns

    """

    filenames = os.listdir(path)

    index_images_to_display = random.sample(range(len(filenames)), nbr_images_to_display)

    

    # List of image names

    list_images = []

    for i in index_images_to_display:

        filename = os.listdir(path)[i]

        img = cv2.imread(path+'/'+filename,cv2.IMREAD_COLOR)

        list_images.append(img)

    

    # Plot figure

    fig=plt.figure(figsize=(8, 8))

    for i in range(nbr_images_to_display):

        img = list_images[i]

        fig.add_subplot(rows, columns, i+1)

        plt.imshow(img)

    plt.show()

        

display_images(path_daisy, 10, 5, 2)
# Check filenames extensions

def extension_unique_values(path):

    extensions = [os.listdir(path)[i][-3:] for i in range(len(os.listdir(path)))]

    return set(extensions)
print('daisy: {}'.format(extension_unique_values(path_daisy)))

print('dandelion: {}'.format(extension_unique_values(path_dandelion)))

print('rose: {}'.format(extension_unique_values(path_rose)))

print('sunflower: {}'.format(extension_unique_values(path_sunflower)))

print('tulip: {}'.format(extension_unique_values(path_daisy)))

filelist_py = [ f for f in os.listdir(path_dandelion) if f.endswith(".py")]

filelist_pyc = [ f for f in os.listdir(path_dandelion) if f.endswith(".pyc")]

filelist_py, filelist_pyc
# Delete files that are not images

def clean_dandelion():

    filelist_py = [ f for f in os.listdir(path_dandelion) if f.endswith(".py")]

    filelist_pyc = [ f for f in os.listdir(path_dandelion) if f.endswith(".pyc")]

    for f in filelist_py + filelist_pyc:

        os.remove(os.path.join(path_dandelion, f))



# Delete useless files

clean_dandelion()    
filelist_py = [ f for f in os.listdir(path_dandelion) if f.endswith(".py")]

filelist_pyc = [ f for f in os.listdir(path_dandelion) if f.endswith(".pyc")]

filelist_py, filelist_pyc 
# Create X and y

images = []

labels = []

img_size = 256



def make_data(flower_type, path):

    

    for filename in os.listdir(path):

        label = flower_type

        path_image = os.path.join(path, filename)

        img = np.array(cv2.imread(path_image, cv2.IMREAD_COLOR)).astype('float32')

        img = cv2.resize(img, (img_size, img_size))

        images.append(img)

        labels.append(label)     

        

make_data("daisy", path_daisy)

make_data("dandelion", path_dandelion)

make_data("rose", path_rose)

make_data("sunflower", path_sunflower)

make_data("tulip", path_tulip)
# Check size compatibility

len(labels), len(images)
# Encode labels

Y = LabelEncoder().fit_transform(labels)

y = to_categorical(Y,5)

# Normalize

X = np.array(images)/255
X.shape
# Split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)
K.clear_session()
model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(256,256,3))

for layer in model.layers[:5]:

    layer.trainable = False
# Adding custom Layers 

x = model.output

x = Flatten()(x)

x = Dense(512, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(512, activation="relu")(x)

predictions = Dense(5, activation="softmax")(x)
# creating the final model 

modelFlower = Model(input = model.input, output = predictions)
# compile the model 

modelFlower.compile(loss = "categorical_crossentropy", optimizer = SGD(lr=0.0001, momentum=0.9), metrics=["categorical_accuracy"])
batch_size = 150

epochs = 15

modelFlower.fit(X_train, y_train, epochs=epochs,batch_size=batch_size, validation_data = (X_test,y_test))
# serialize model to JSON

model_json = modelFlower.to_json()

with open("modelFlower.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

modelFlower.save_weights("modelFlower.h5")

print("Saved model to disk")