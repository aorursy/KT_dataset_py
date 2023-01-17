import os

from os import listdir

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from matplotlib.pyplot import figure, imshow, axis

from matplotlib.image import imread



import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

import pandas as pd

import seaborn as sns

from tqdm import tqdm



import keras

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD
labels = pd.read_csv('../input/train.csv')

test_path = "../input/test/"

train_path = "../input/train/"



colors = ["blue", "green", "red", "yellow"] 
labels.head()
test = []

for file in listdir(test_path):

    fname = file.split("_")[0]

    test.append(fname)
def get_label(label):

    

    num = list(map(int, label.split()))



    return np.eye(28, dtype=np.float)[num].sum(axis=0) # convert an array to one-hot coded then sum along the columns
# display 4 channels of an image id

def showImagesHorizontally(file_num):

    # file_num is a list of integers

    fname = f"{train_path}{labels.Id[file_num]}_"

    

    fig = figure(figsize=(15,5))

    imgs = [fname + x + ".png" for x in colors]

    

    for i in range(len(colors)):

        a = fig.add_subplot(1, 4, i+1)

        img = plt.imread(imgs[i])

        plt.title(f'{colors[i]}')

        plt.imshow(img)

        axis('off')

        

showImagesHorizontally(5)
def rgby_generator(id):

    im_blue = imread(f"{train_path}{id}_blue.png")

    im_green = imread(f"{train_path}{id}_green.png")

    im_red = imread(f"{train_path}{id}_red.png")

    im_yellow = imread(f"{train_path}{id}_yellow.png")

    

    rgby = np.stack((im_red, im_green, im_blue, im_yellow),-1)

    

    return rgby
def ConvBlock(layers, model, filters):

    for i in range(layers): 

        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2,2), strides=(2,2)))
def VGG16():  

    # initialize the model

    model = Sequential()



    # input layer

    model.add(Conv2D(64, (3, 3), input_shape=(512, 512, 4), activation='relu', padding='same'))

    

    # Conv Block 1

    ConvBlock(1, model, 64)



    # Conv Block 2

    ConvBlock(2, model, 128)



    # Conv Block 3

    ConvBlock(3, model, 256)



    # Conv Block 4

    ConvBlock(3, model, 512)



    # Conv Block 5

    ConvBlock(3, model, 512)



    # FC layers

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))

    model.add(Dense(4096, activation='relu'))

    model.add(Dense(28, activation='softmax'))

    

    return model
model = VGG16()



# Compile model

model.compile(optimizer= optimizers.Adam(), loss= 'binary_crossentropy', metrics= ['acc'])
x_train = [rgby_generator(labels.Id[i]) for i in tqdm(range(100))]

y_train = [get_label(y) for y in labels.Target[0:100]]



x_train = np.array(x_train)

y_train = np.array(y_train)
model.fit(x_train, y_train, epochs=5)