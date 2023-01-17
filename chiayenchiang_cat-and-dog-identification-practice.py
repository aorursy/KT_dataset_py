# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import cv2   #use it in reading and resizing our Images.

import numpy as np  #process large, multi-dimensional arrays and matrices super easy and fast.

import pandas as pd #manipulating numerical tables and time series.

import matplotlib.pyplot as plt #plotting lines, bar-chart, graphs, histograms

%matplotlib inline 

#makes our plots appear in the notebook



import os #accessing your computer and file system.

import random # create random numbers, split or shuffle our data set.

import gc # garbage collector is an important tool for manually cleaning and deleting unnecessary variables.
#create a file path for both test and train sets

train_dir = '../input/train'

test_dir = '../input/test'



#create two variables train_dogs and train_cats

#write a list comprehension: os.listdir() to get all the images in the train data

#and retrieve all images with dog/cat in their name

train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]

train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]



#get our test images

test_imgs = ['../input/test/{}'.format(i) for i in os.listdir(test_dir) ]



# with little computational power, we’re going to extract only 2000 images for both classes

train_imgs = train_dogs[:2000] + train_cats[:2000]

random.shuffle(train_imgs) #randomly shuffle the train_imgs



# delete two columns to save memories

del train_dogs

del train_cats

gc.collect()
#Import an image plotting module from matplotlib

import matplotlib.image as mpimg

#Run a for loop to plot the first three images in train_imgs

for ima in train_imgs[0:3]:

    img = mpimg.imread(ima)

    imgplot = plt.imshow(img)

    plt.show()
#resize the images using the cv2 module

#declare the new dimensions we want to use: 150 by 150 for height and width and 3 channels



nrows = 150

ncolumns = 150

channels = 3 #change to 1 if you want to use grayscale image

def read_and_process_image(list_of_images):

    """

    Returns two arrays:

        X is an array of resized images

        y is an array of labels

    """

    X = [] #images

    y = [] #labels

    

    for image in list_of_images: #read images one after the other and resize them with the cv2 commands.

        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation = cv2.INTER_CUBIC)) #read the image

        #get the labels

        if 'dog' in image:

            y.append(1)

        elif 'cat' in image:

            y.append(0)

        

    return X, y

            
X, y = read_and_process_image(train_imgs)
X[0]
y
#We can’t plot the images in X with the mpimg module of matplotlib.image above 

#because these are now arrays of pixels not raw jpg files

#So we should use the imshow() command.





plt.figure(figsize = (20, 10))

columns = 5 

for i in range(columns):

    plt.subplot(5/ columns + 1, columns, i + 1)

    plt.imshow(X[i])
import seaborn as sns



#we delete the train_imgs, since it has already been converted to an array and saved in X.

del train_imgs

gc.collect()



#X and y are currently of type list (list of python array)

#convert list to numpy array so we can feed it into our model

X = np.array(X)

y = np.array(y)



#Lets plot the to be sure we just have two classes

#Plot a colorful diagram to confirm the number of classes

sns.countplot(y)

plt.title('Labels for Cats and Dogs')
#check the shape of data 

print("Shape of the image is:", X.shape)

print("Shape of the label is:", y.shape)



#keras model takes as input an array of (height, width,channels)
#split data into test and train set

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 2)



print("Shape of train images is", X_train.shape)

print("Shape of validation images is", X_val.shape)

print("Shape of train label is", y_train.shape)

print("Shape of validation label is", y_val.shape)
del X

del y

gc.collect()



#get the length of the train and validation data

ntrain = len(X_train)

nval = len(X_val)



# use batch size of 32

batch_size = 32



#use a Convolutional Neural Network (convnet) to train our model.

#In creating our model we’re going to use KERAS.



#Keras is an open source neural network library written in Python. 

#It is capable of running on top of TensorFlow
from keras import layers

from keras import models #Sequential model will be used

from keras import optimizers #contains different types of back propagation algorithm for training our model

from keras.preprocessing.image import ImageDataGenerator #(ImageDataGenerator) used when working with a small data set

from keras.preprocessing.image import img_to_array, load_img

#create our Network architecture follow VGGnet structure

#32 > 64 > 128 > 512 > 1
#create a sequential model so that 

#tells keras to stack all layers sequentially

model = models.Sequential() 



# Conv2D (filter size,  kernel_size,  activation function, input shape)

# filter size: the size of the output dimension

# kernel_size: the height and width of the 2D convolution window.

# input shape : the dimensions we resized our images (We do not pass 4000 since it's  the batch dimension.)

model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape =(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3,3), activation = 'relu'))



#add a MaxPool2D layer 

#to reduce the spatial size of the incoming features

#thereby helping to reduce overfitting.

model.add(layers.MaxPooling2D((2, 2)))



#How A conv2D layers extract and learn spatial features:

#(1) it has been flattened

#(2) then passed to a dense layer 

model.add(layers.Flatten())



# add a Dropout layer: randomly drop half of the layers. (with value = 0.5)

# therefore, the network learns to be independent and not reliable on a single layer.

model.add(layers.Dropout(0.5)) 



model.add(layers.Dense(512, activation ='relu'))

model.add(layers.Dense(1, activation ='sigmoid')) # sigmoid function in the end because we have just two classes

model.summary()
model.compile(loss ='binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4), metrics = ['acc'])

# specify a loss function: optimizer will minimize the cost; and since it is two class problem, we use binary crossentropy loss 

# use one of the optimizers called rmsprop: calculate the difference between a world class model and a naive one

# specify which metric we want to use: to know if our model is doing well, and since it is a classification problem

# , the accuracy metric (acc) is a good choice. 
#perform some Normalization. 

#i.e scale our image pixel values to have a unit standard deviation and a mean of 0.
#ImageDataGenerator() automatically turn image files into preprocessed tensors 

#that can be fed directly into models during trainng.

# is able to:

#  Decode the JPEG to RGB grids

#  Convert these into floating-point tensors

#  rescale pixel values (between 0 and 255) to the [0, 1] interval 

#  easily augment images:  important feature for small training set



# augmentation configuration: prevent overfitting since we have small training set

train_datagen = ImageDataGenerator(rescale = 1./255, # scale the image between 0 and 1

                                  rotation_range = 0.2,

                                  width_shift_range = 0.2,

                                  height_shift_range = 0.2,

                                  shear_range = 0.2,

                                  zoom_range = 0.2,

                                  horizontal_flip = True,)



val_datagen = ImageDataGenerator(rescale = 1./255) # do not augment val_dat, only need rescale 
# create two generators for both training and validation

train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)

val_generator = val_datagen.flow(X_val, y_val, batch_size = batch_size)
#

history = model.fit_generator(train_generator, 

                             steps_per_epoch = ntrain // batch_size,

                             epochs = 64, 

                             validation_data = val_generator,

                             validation_steps = nval // batch_size)
model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')