# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import glob

import h5py

import shutil

import imgaug as aug

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mimg

import imgaug.augmenters as iaa

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

from pathlib import Path

from skimage.io import imread

from skimage.transform import resize

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

import cv2

from keras import backend as K

color = sns.color_palette()

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))
import tensorflow as tf



# Set the seed for hash based operations in python

os.environ['PYTHONHASHSEED'] = '0'



# Set the numpy seed

np.random.seed(111)



# Disable multi-threading in tensorflow ops

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)



# Set the random seed in tensorflow at graph level

tf.set_random_seed(111)



# Define a tensorflow session with above session configs

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)



# Set the session in keras

K.set_session(sess)



# Make the augmentation sequence deterministic

aug.seed(111)
data_dir = Path('../input/New folder (2)')



# Path to train directory (Fancy pathlib...no more os.path!!)

train_dir = data_dir / 'train'



# Path to validation directory

val_dir = data_dir / 'valid'



# Path to test directory

test_dir = data_dir / 'test'
normal_cases_dir = train_dir / 'NORMAL'

pneumonia_cases_dir = train_dir / 'PNEUMONIA'



# Get the list of all the images

normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



# An empty list. We will insert the data into this list in (img_path, label) format

train_data = []



# Go through all the normal cases. The label for these cases will be 0

for img in normal_cases:

    train_data.append((img,0))



# Go through all the pneumonia cases. The label for these cases will be 1

for img in pneumonia_cases:

    train_data.append((img, 1))



# Get a pandas dataframe from the data we have in our list 

train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)



# Shuffle the data 

train_data = train_data.sample(frac=1.).reset_index(drop=True)



# How the dataframe looks like?

train_data.head()
cases_count = train_data['label'].value_counts()

print(cases_count)



# Plot the results 

plt.figure(figsize=(10,8))

sns.barplot(x=cases_count.index, y= cases_count.values)

plt.title('Number of cases', fontsize=14)

plt.xlabel('Case type', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])

plt.show()
normal_cases_dir = val_dir / 'NORMAL'

pneumonia_cases_dir = val_dir / 'PNEUMONIA'



# Get the list of all the images

normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



# List that are going to contain validation images data and the corresponding labels

valid_data = []

valid_labels = []







# Some images are in grayscale while majority of them contains 3 channels. So, if the image is grayscale, we will convert into a image with 3 channels.

# We will normalize the pixel values and resizing all the images to 224x224 



# Normal cases

for img in normal_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(0, num_classes=2)

    valid_data.append(img)

    valid_labels.append(label)

    

                      

# Pneumonia cases        

for img in pneumonia_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    #mean_img = np.mean(img, axis=0)

    #std_img = np.std(img, axis=0)

    #x_train_norm = (img - mean_img) / std_img

    label = to_categorical(1, num_classes=2)

    valid_data.append(img)

    valid_labels.append(label)

    

# Convert the list into numpy arrays

valid_data = np.array(valid_data)

valid_labels = np.array(valid_labels)



print("Total number of validation examples: ", valid_data.shape)

print("Total number of labels:", valid_labels.shape)
seq = iaa.OneOf([

    iaa.Fliplr(), # horizontal flips

    iaa.Affine(rotate=22), # roatation

    iaa.Multiply((1.2, 1.5)),

    iaa.Affine(

        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},

        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},

        rotate=(-25, 25),

        shear=(-8, 8)

    )

]) 
def data_gen(data, batch_size):

    # Get total number of samples in the data

    n = len(data)

    steps = n//batch_size

    

    # Define two numpy arrays for containing batch data and labels

    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)

    batch_labels = np.zeros((batch_size,2), dtype=np.float32)



    # Get a numpy array of all the indices of the input data

    indices = np.arange(n)

    

    # Initialize a counter

    i =0

    while True:

        np.random.shuffle(indices)

        # Get the next batch 

        count = 0

        next_batch = indices[(i*batch_size):(i+1)*batch_size]

        for j, idx in enumerate(next_batch):

            img_name = data.iloc[idx]['image']

            label = data.iloc[idx]['label']

            

            # one hot encoding

            encoded_label = to_categorical(label, num_classes=2)

            # read the image and resize

            img = cv2.imread(str(img_name))

            img = cv2.resize(img, (224,224))

            

            # check if it's grayscale

            if img.shape[2]==1:

                img = np.dstack([img, img, img])

            

            # cv2 reads in BGR mode by default

            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # normalize the image pixels

            orig_img = img.astype(np.float32)/255.

            #mean_img = np.mean(orig_img, axis=0)

            #std_img = np.std(orig_img, axis=0)

            #x_train_norm = (orig_img - mean_img) / std_img

            

            batch_data[count] = orig_img

            batch_labels[count] = encoded_label

            

            # generating more samples of the undersampled class

            if label==0 and count < batch_size-2:

                aug_img1 = seq.augment_image(img)

                aug_img2 = seq.augment_image(img)

                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)

                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)

                aug_img1 = aug_img1.astype(np.float32)/255.

                aug_img2 = aug_img2.astype(np.float32)/255.



                batch_data[count+1] = aug_img1

                batch_labels[count+1] = encoded_label

                batch_data[count+2] = aug_img2

                batch_labels[count+2] = encoded_label

                count +=2

            

            else:

                count+=1

            

            if count==batch_size-1:

                break

            

        i+=1

        yield batch_data, batch_labels

            

        if i>=steps:

            i=0
batch_size = 16

nb_epochs = 1



# Get a train data generator

train_data_gen = data_gen(data=train_data, batch_size=batch_size)



# Define the number of training steps

nb_train_steps = train_data.shape[0]//batch_size



print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(valid_data)))
image_size = 224

IMG_SHAPE = (image_size, image_size, 3)



# Create the base model from the pre-trained model VGG16

base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,

                                               include_top=False,

                                               weights='imagenet')
base_model.trainable = True
fine_tune = 15

for layer in base_model.layers[:fine_tune]:

  layer.trainable =  False
from tensorflow.keras import models

model = models.Sequential()

#model.add(tf.keras.layers.BatchNormalization(input_shape=IMG_SHAPE,momentum=0.1))

model.add(base_model)

model.add(tf.keras.layers.Flatten())

#model.add(tf.keras.layers.Dropout(0.2))

#x2=model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(128, activation='relu'))

#model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(64, activation='relu'))

#model.add(tf.keras.layers.BatchNormalization(momentum=0.7))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(lr=1e-4),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
x=5

history = model.fit_generator(train_data_gen, epochs=x, steps_per_epoch=nb_train_steps,

                               

                               validation_data=(valid_data, valid_labels)

                              )
normal_cases_dir = test_dir / 'NORMAL'

pneumonia_cases_dir = test_dir / 'PNEUMONIA'



normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



test_data = []

test_labels = []



for img in normal_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(0, num_classes=2)

    test_data.append(img)

    test_labels.append(label)

                      

for img in pneumonia_cases:

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    else:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(1, num_classes=2)

    test_data.append(img)

    test_labels.append(label)

    



test_data = np.array(test_data)

test_labels = np.array(test_labels)



print("Total number of test examples: ", test_data.shape)

print("Total number of labels:", test_labels.shape)
test_loss, test_score = model.evaluate(test_data, test_labels, batch_size=16)

print("Loss on test set: ", test_loss)

print("Accuracy on test set: ", test_score)
preds = model.predict(test_data, batch_size=16)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(test_labels, axis=-1)

cm  = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
tn, fp, fn, tp = cm.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)

f1= 2/((1/precision)+(1/recall))



print("Recall of the model is {:.2f}".format(recall))

print("Precision of the model is {:.2f}".format(precision))

print("F1 score of the model is {:.2f}".format(f1))
