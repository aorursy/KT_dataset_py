import numpy as np 

import pandas as pd 

import cv2 # image and video processing library to be used for reading and resizing our images



import matplotlib.pyplot as plt

%matplotlib inline 



import random # to split and shuffle our dataset

import gc # garbage collector for cleaning deleted data from memory



# Input data files are available in the "../input/" directory.



import os

'''for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))'''



# Any results you write to the current directory are saved as output.
# create a file path to our train and test data

train_dir = '../input/dogs-vs-cats-redux-kernels-edition/train'

test_dir = '../input/dogs-vs-cats-redux-kernels-edition/test'
# list comprehension to get all the images in the train data zip file

# and retrieve all images with dog/cat in their name

train_dogs = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i] # get dog images

train_cats = ['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i] # get cat images



test_imgs = ['../input/dogs-vs-cats-redux-kernels-edition/test/{}'.format(i) for i in os.listdir(test_dir)] # get test images
train_imgs = train_dogs[:2000] + train_cats[:2000] # grab the images from train_dogs and train_cats and concatenate them
random.shuffle(train_imgs) #shuffle the training images so they aren't ordered by first half dogs and second half cats
# now that we have train_imgs, we don't need train_dogs and train_cats so we get rid of them so we don't run out of 

# memory when training our model

del train_dogs

del train_cats

gc.collect()
# lets view some images in train_imgs

import matplotlib.image as mpimg #import an image plotting module from matplotlib

for ima in train_imgs[0:3]: # run a for loop to plot the first three images in train_imgs

    img = mpimg.imread(ima)

    imgplot = plt.imshow(img)

    plt.show()
# resize the images using the cv2 module

nrows = 150

ncolumns = 150

channels = 3
# function to read and resize the images

def read_and_process_image(list_of_images):

    '''

    Returns two arrays:

    X is an array of resized images

    y is an array of labels

    '''

    X = [] # images

    y = [] # labels

    

    for image in list_of_images:

        # read the image

        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))



        # get the labels

        if 'dog' in image[-13:-1]:

            y.append(1)

        elif 'cat' in image[-13:-1]:

            y.append(0)

        

    return X, y
# call the function

X, y = read_and_process_image(train_imgs)
# preview of the first image and label

X[0]
y[0:5]
plt.figure(figsize=(20,10))

columns = 5

for i in range(columns):

    plt.subplot(5/columns + 1, columns, i+1)

    plt.imshow(X[i])
import seaborn as sns

del train_imgs # delete train_imgs since it has already been converted to an array and saved in X

gc.collect()



# convert list to numpy array to use in our model as X and y are currently a python array list

X = np.array(X)

y = np.array(y)



# lets make sure our labels contain the correct number of images

sns.countplot(y)

plt.title('Labels for Cats and Dogs')
# check shape of data

print('Shape of train images is:', X.shape)

print('Shape of labels is:', y.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2) 

# test_size =0.2 means we set 20% of the data to be assigned to the validation set, 

# and 80% to the train set
print('Shape of train images is:', X_train.shape)

print('Shape of test images is:', X_val.shape)

print('Shape of labels is:', y_train.shape)

print('Shape of labels is:', y_val.shape)
del X

del y

gc.collect()
# get the length of the train and validation data

ntrain = len(X_train)

ntest = len(X_val)



batch_size = 32
from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img
model = models.Sequential() 

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))) 

model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
# we'll use the RMSprop optimizer with a learning rate of 0.0001

# we'll use binary_crossentropy loss because its a binary classification

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
train_datagen = ImageDataGenerator(rescale=1./255, 

                                  rotation_range=40,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)
# with the ImageDataGenerator complete, pass the train and validation set

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
# training the model

history = model.fit_generator(train_generator,

                             steps_per_epoch=ntrain // batch_size,

                             epochs = 64,

                             validation_data=val_generator,

                             validation_steps=ntest // batch_size)
# save the model for use next time without training again

model.save_weights('model_weights.h5')

model.save('model_keras.h5')
# prediction on the first ten images of the test dataset

X_test, y_test = read_and_process_image(test_imgs[0:10])

x = np.array(X_test)

test_datagen = ImageDataGenerator(rescale=1./255)
i = 0

text_labels = [] # create a list to hold the labels we will generate

plt.figure(figsize=(30,20)) # figure size of the images we will plot

for batch in test_datagen.flow(x, batch_size=1):

    pred = model.predict(batch) # make a prediction on the image 

    if pred > 0.5: # if prediction is > 0.5 it should be a dog, so append 'dog'

        text_labels.append('dog')

    else: # otherwise, it should be a cat

        text_labels.append('cat')

    plt.subplot(5 / columns + 1, columns, i + 1) # subplot to plot multiple images

    plt.title('This is a ' + text_labels[i])

    imgplot = plt.imshow(batch[0])

    i+=1

    if i % 10 == 0:

        break

plt.show()