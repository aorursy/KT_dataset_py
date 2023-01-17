# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import cv2  # It is used for all sorts of image and video analysis, like facial recognition and detection, 

            # license plate reading, photo editing, advanced robotic vision, optical character recognition, and a whole lot more..

import numpy as np # linear algebra

                   # It makes working and computing large, multi-dimensional arrays and matrices super easy and fast.

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #  It can be used for plotting lines, bar-chart, graphs, histograms and even displaying Images.

# %matplotlib inline # makes our plots appear in the notebook.

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

import seaborn as sns

import os

import random

import gc #short for garbage collector is an important tool for manually cleaning and deleting unnecessary variables. 



# Necessary keras module

from keras import layers

# Here we import keras layers module which contains different types of layers used in deep learning such as:

# ** Convolutional layer (Mostly used in computer vision)

# ** Pooling layer (also used in computer vision)

# ** Recurrent layer (Mostly used in sequential and time series modelling)

# ** Embedding layers (Mostly used in Natural Language processing)

# ** Normalization layers 

# ** and many more

from keras import models

from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation

from keras import optimizers

# Here we import keras optimizer, a module that contains different types of back propagation algorithm for training our model. Some of these optimizers are:

# **sgd (stochastic gradient descent)

# **rmsprop (root mean square propagation)

# **Adams

# **Adagrad

# **Adadelta



from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img





print(os.listdir("../input"))



train_dir = '../input/train'

test_dir = '../input/test'



train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i] # get dog images

train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i] # get cat images

test_images = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)] # get test images



train_images = train_dogs[:2000] + train_cats[:2000]

random.shuffle(train_images)

del(train_dogs)

del(train_cats)

gc.collect()



# for ima in train_images[0:3]:

#     img = mpimg.imread(ima)

#     imaplot = plt.imshow(img)

#     plt.show()

    

nrows = 150

ncolumns = 150

channels = 3



# A function to read and process the image to an acceptable format for our model

def read_and_process_images(list_of_images):

    """

    returns two arrays

    x = is an array of resized image

    y = is an array of labels

    """

    

    x = [] # images

    y = [] # labels

    

    for image in list_of_images:

        x.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)) # cv2.resize(src, dim, interpolation = {})

        if 'dog' in image:

            y.append(1)

        if 'cat' in image:

            y.append(0)

    return x, y

X, Y = read_and_process_images(train_images)

plt.figure(figsize=(20,10))

columns = 5

for i in range(columns):

    plt.subplot(5/columns+1, columns, i+1)

    plt.imshow(X[i])

    

del train_images

gc.collect()

#convert a list to numpy array

X = np.array(X)

Y = np.array(Y)

sns.countplot(Y)

plt.title('Cats and Dogs')

print("Shape of X : ", X.shape)

print("Shape of Y : ", Y.shape)



X_train, X_val,  Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=2)

print("Shape of X_Train : ", X_train.shape)

print("Shape of Y_train : ", Y_train.shape)

print("Shape of X_Val : ", X_val.shape)

print("Shape of Y_Val : ", Y_val.shape)



del X

del Y

gc.collect()



# get the length of the train and val data

ntrain = len(X_train)

nval = len(X_val)



batch_size = 32



# Model Creation

optimizer = optimizers.RMSprop(lr=1e-4)

objective = 'binary_crossentropy'



def cats_and_dogs_model():

    model = models.Sequential();

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3,3), activation='relu'))

#     model.add(layers.MaxPooling2D((2,2)))

#     model.add(layers.Conv2D(512, (3,3), activation='relu'))

    model.add(Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['acc'])

    model.summary()

    return model



model = cats_and_dogs_model()



# Lets create the augmentation configuration

# This helps preventing overfitting as we are using small datasets

train_datagen = ImageDataGenerator(rescale=1./255, #Scale the image between 0 and 1

#                                    We pass the rescale option to the ImageDataGenerator object. 

#                                    The rescale=1./255 option is a very IMPORTANT parameter. 

#                                    It normalizes the image pixel values to have zero mean and

#                                    standard deviation of 1.

#                                    It helps your model to generally learn and update its parameters efficiently.

                                   rotation_range = 40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True

)



val_datagen = ImageDataGenerator(rescale=1./255)

print(train_datagen)



#Create the Image Generators

train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size)

val_generator = val_datagen.flow(X_val, Y_val, batch_size=batch_size)



print(ntrain, " ", nval)

# Train the model

history = model.fit_generator(train_generator,

                              steps_per_epoch=ntrain/batch_size,

#                               Here we specify the number of steps per epoch. 

#                               This tells our model how many images we want to process before making a 

#                               gradient update to our loss function.

#                               A total of 3200 images divided by batch size of 32 will give us 100 steps. 

#                               This means we going to make a total of 100 gradient update to our model in one pass through the entire training set.

                              epochs=100,

#                               An epoch is a full cycle or pass through the entire training set. 

#                               In our case, an epoch is reached when we make 100 gradient updates as specified by our 

#                               steps_per_epoch parameter.

#                               Epochs = 64, means we want to go over our training data 64 times and 

#                               each time we will make gradient updates 100 times.

                              validation_data=val_generator,

                              validation_steps=nval/batch_size

                             )



# Any results you write to the current directory are saved as output.
# Lets plot the train and val curve

# get the details from the history object

# After training a keras model, it always calculates and saves the metric 

# we specified when we compiled our model in a variable called history. 

# We can extract these values and plot them.

# Note: The history object contains all the updates that happened during training.

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



# Here we simply get the size of our epoch from the number of values in the ‘acc’ list.

epochs = range(1, len(acc)+1)



# Train and validation accuracy

# Here we plot the accuracy against the epoch size.

plt.plot(epochs, acc, 'b', label='Training Accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')

plt.title("Training and Validation Accuracy")

plt.legend

plt.figure







#Train and Validation Loss

plt.plot(epochs, loss, 'b', label='Trainning Loss')

plt.plot(epochs, val_loss, 'r', label='Validation Loss')

plt.title("Training and Validation Loss")

plt.legend

plt.show
# Now lets predict on the first 10 images of test data

x_test, y_test = read_and_process_images(test_images[0:10])

x = np.array(x_test)

test_datagen = ImageDataGenerator(rescale=1./255)

x
i=0

text_labels = []

plt.figure(figsize=(30,20))

for batch in test_datagen.flow(x, batch_size=1):

    pred = model.predict(batch)

    if pred>0.5:

        text_labels.append('dog')

    else:

        text_labels.append('cat')

    plt.subplot(5/columns+1, columns, i+1)

    plt.title("this is a " + text_labels[i])

    imgplot = plt.imshow(batch[0])

    

    print(pred)

    i = i+1;

    if i&10 == 0:

        break

plt.show()