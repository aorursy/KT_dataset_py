# Author : Rajaraman

# project : Recipe Classifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import glob

from tqdm import tqdm

from skimage import io, transform

from keras.utils import to_categorical

import time

import warnings



def fxn():

    warnings.warn("deprecated", DeprecationWarning)



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    fxn()

from sklearn.model_selection import train_test_split

seed = 333

np.random.seed(seed)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#path to images

img_dir = "../input/gastrointestinal/kvasir-dataset/"



#list all available images type

DATA_DIR = os.listdir(img_dir)

print(DATA_DIR)

data = list(enumerate(os.listdir(img_dir)))

for x in data:

    print(x)

# 4, 0, 2, 7
print(img_dir)

def load_data(img_dir):

    X = []

    y = []

    labels = []

    idx = 0

    for i,folder_name in data:

        if folder_name in DATA_DIR:

            labels.append(folder_name)

            for file_name in tqdm(os.listdir(f'{img_dir}/{folder_name}')):

                if file_name.endswith('jpg'):

                    im = io.imread(f'{img_dir}/{folder_name}/{file_name}')

                    if im is not None:

                        im = transform.resize(im, (100, 100))

                        X.append(im)

                        y.append(idx)



        idx+=1

    X = np.asarray(X)

    y = np.asarray(y)

    labels = np.asarray(labels)

    return X,y,labels
X,y,labels = load_data(img_dir)
print(X)

print(y)

print(labels)
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
train_img = X_train

train_labels = y_train

test_img = X_test

test_labels = y_test

train_img.shape, train_labels.shape, test_img.shape, test_labels.shape
#show random samples

rand_14 = np.random.randint(0, train_img.shape[0],48)

sample_img = train_img[rand_14]

sample_labels = train_labels[rand_14]

num_rows, num_cols = 2, 24

f, ax = plt.subplots(num_rows, num_cols, figsize=(12,5),gridspec_kw={'wspace':0.03, 'hspace':0.01})

for r in range(num_rows):

    for c in range(num_cols):

        image_index = r * 7 + c

        ax[r,c].axis("off")

        ax[r,c].imshow(sample_img[image_index])

        ax[r,c].set_title('%s' % sample_labels[image_index])

plt.show()

plt.close()
#one-hot-encode the labels

num_classes = len(labels)

train_labels_cat = to_categorical(train_labels,num_classes)

test_labels_cat = to_categorical(test_labels,num_classes)

train_labels_cat.shape, test_labels_cat.shape
# re-shape the images data

train_data = train_img

test_data = test_img

train_data.shape, test_data.shape
# shuffle the training dataset & set aside val_perc % of rows as validation data

for _ in range(5): 

    indexes = np.random.permutation(len(train_data))



# randomly sorted!

train_data = train_data[indexes]

train_labels_cat = train_labels_cat[indexes]



# now we will set-aside val_perc% of the train_data/labels as cross-validation sets

val_perc = 0.10

val_count = int(val_perc * len(train_data))

print(val_count)



# first pick validation set

val_data = train_data[:val_count,:]

val_labels_cat = train_labels_cat[:val_count,:]



# leave rest in training set

train_data2 = train_data[val_count:,:]

train_labels_cat2 = train_labels_cat[val_count:,:]



train_data2.shape, train_labels_cat2.shape, val_data.shape, val_labels_cat.shape, test_data.shape, test_labels_cat.shape
 # a utility function that plots the losses and accuracies for training & validation sets across our epochs

def show_plots(history):

    """ Useful function to view plot of loss values & accuracies across the various epochs """

    loss_vals = history['loss']

    val_loss_vals = history['val_loss']

    epochs = range(1, len(history['acc'])+1)

    

    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(16,4))

    

    # plot losses on ax[0]

    ax[0].plot(epochs, loss_vals, color='navy',marker='o', linestyle=' ', label='Training Loss')

    ax[0].plot(epochs, val_loss_vals, color='firebrick', marker='*', label='Validation Loss')

    ax[0].set_title('Training & Validation Loss')

    ax[0].set_xlabel('Epochs')

    ax[0].set_ylabel('Loss')

    ax[0].legend(loc='best')

    ax[0].grid(True)

    

    # plot accuracies

    acc_vals = history['acc']

    val_acc_vals = history['val_acc']



    ax[1].plot(epochs, acc_vals, color='navy', marker='o', ls=' ', label='Training Accuracy')

    ax[1].plot(epochs, val_acc_vals, color='firebrick', marker='*', label='Validation Accuracy')

    ax[1].set_title('Training & Validation Accuracy')

    ax[1].set_xlabel('Epochs')

    ax[1].set_ylabel('Accuracy')

    ax[1].legend(loc='best')

    ax[1].grid(True)

    

    plt.show()

    plt.close()

    

    # delete locals from heap before exiting

    del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals 
def print_time_taken(start_time, end_time):

    secs_elapsed = end_time - start_time

    

    SECS_PER_MIN = 60

    SECS_PER_HR  = 60 * SECS_PER_MIN

    

    hrs_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_HR)

    mins_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_MIN)

    

    if hrs_elapsed > 0:

        print('Time taken: %d hrs %d mins %d secs' % (hrs_elapsed, mins_elapsed, secs_elapsed))

    elif mins_elapsed > 0:

        print('Time taken: %d mins %d secs' % (mins_elapsed, secs_elapsed))

    elif secs_elapsed > 1:

        print('Time taken: %d secs' % (secs_elapsed))

    else:

        print('Time taken - less than 1 sec')
labels
def get_commonname(idx):

    sciname = labels[idx][0]

    return {

        'normal-z-line':'normal-z-line',

        'ulcerative-colitis':'ulcerative-colitis',

        'dyed-resection-margins':'dyed-resection-margins',

        'esophagitis': 'dyed-lifted-polyps',

        'dyed-lifted-polyps': 'dyed-lifted-polyps',

        'normal-pylorus': 'normal-pylorus',

        'normal-cecum': 'normal-cecum',

        'polyps': 'polyps'

    }[sciname]
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout





import numpy as np

from keras.utils.np_utils import to_categorical



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,Activation,MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.datasets import mnist

from keras.models import load_model

from sklearn.model_selection import train_test_split

from keras.utils import np_utils

from PIL import Image
#data augmentation

datagen = ImageDataGenerator(

        rotation_range=30,

        zoom_range = 0.25,  

        width_shift_range=0.1, 

        height_shift_range=0.1)

# datagen = ImageDataGenerator(

#     rotation_range=8,

#     shear_range=0.3,

#     zoom_range = 0.08,

#     width_shift_range=0.08,

#     height_shift_range=0.08)
#create multiple cnn model for ensembling

#model 1

model = Sequential()



model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (100, 100, 3)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(128, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))





model.add(Conv2D(256, kernel_size = 4, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))



# use adam optimizer and categorical cross entropy cost

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# after each epoch decrease learning rate by 0.95

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)



# train

epochs = 100

j=0

start_time = time.time()

history = model.fit_generator(datagen.flow(train_data2, train_labels_cat2, batch_size=64),epochs = epochs, steps_per_epoch = train_data2.shape[0]/64,validation_data = (val_data, val_labels_cat), callbacks=[annealer], verbose=1)

end_time = time.time()

print_time_taken(start_time, end_time)





print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(j+1,epochs,history.history['acc'][epochs-1],history.history['val_acc'][epochs-1]))
show_plots(history.history)
test_loss, test_accuracy = model.evaluate(test_data, test_labels_cat, batch_size=64)

print('Test loss: %.4f accuracy: %.4f' % (test_loss, test_accuracy))
im_list = [6, 121]

for i in im_list:

#     i = 1000  #index from test data to be used, change this other value to see a different image

    img = test_data[i]

    plt.imshow(img)

    plt.show()

    pred = model.predict_classes(img.reshape(-1,100,100,3))

    actual =  test_labels[i]

    print(f'actual: {get_commonname(actual)}')

    print(f'predicted: {get_commonname(pred)}')
from keras.models import model_from_json

from keras.models import load_model



# Creates a HDF5 file 'my_model.h5'

model.save('model_gastro.h5')
# Returns a compiled model identical to the previous one

load_model = load_model('model_gastro.h5')
im_list = [0, 121]

for i in im_list:

#     i = 1000  #index from test data to be used, change this other value to see a different image

    img = test_data[i]

    plt.imshow(img)

    plt.show()

    pred = load_model.predict_classes(img.reshape(-1,100,100,3))

    actual =  test_labels[i]

    print(f'actual: {get_commonname(actual)}')

    print(f'predicted: {get_commonname(pred)}')