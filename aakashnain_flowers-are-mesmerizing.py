# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import glob

import shutil

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mimg

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

from pathlib import Path

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Check for the directory and if it doesn't exist, make one.

cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

    

# make the models sub-directory

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)
# Copy the weights from your input files to the cache directory

!cp ../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/
# Define some paths

input_path = Path('../input/flowers-recognition/flowers/')

flowers_path = input_path / 'flowers'
# Each species of flower is contained in a separate folder . Get all the sub directories

flower_types = os.listdir(flowers_path)

print("Types of flowers found: ", len(flower_types))

print("Categories of flowers: ", flower_types)
# In order to keep track of my data details or in order to do some EDA, I always try to 

# get the information in a dataframe. After all, pandas to the rescue!!



# A list that is going to contain tuples: (species of the flower, corresponding image path)

flowers = []



for species in flower_types:

    # Get all the file names

    all_flowers = os.listdir(flowers_path / species)

    # Add them to the list

    for flower in all_flowers:

        flowers.append((species, str(flowers_path /species) + '/' + flower))



# Build a dataframe        

flowers = pd.DataFrame(data=flowers, columns=['category', 'image'], index=None)

flowers.head()
# Let's check how many samples for each category are present

print("Total number of flowers in the dataset: ", len(flowers))

fl_count = flowers['category'].value_counts()

print("Flowers in each category: ")

print(fl_count)
# Let's do some visualization too

plt.figure(figsize=(12,8))

sns.barplot(x=fl_count.index, y=fl_count.values)

plt.title("Flowers count for each category", fontsize=16)

plt.xlabel("Category", fontsize=14)

plt.ylabel("Count", fontsize=14)

plt.show()
# Let's visualize flowers from each category



# A list for storing names of some random samples from each category

random_samples = []



# Get samples fom each category 

for category in fl_count.index:

    samples = flowers['image'][flowers['category'] == category].sample(4).values

    for sample in samples:

        random_samples.append(sample)







# Plot the samples

f, ax = plt.subplots(5,4, figsize=(15,10))

for i,sample in enumerate(random_samples):

    ax[i//4, i%4].imshow(mimg.imread(random_samples[i]))

    ax[i//4, i%4].axis('off')

plt.show()    
# Make a parent directory `data` and two sub directories `train` and `valid`

%mkdir -p data/train

%mkdir -p data/valid



# Inside the train and validation sub=directories, make sub-directories for each catgeory

%cd data

%mkdir -p train/daisy

%mkdir -p train/tulip

%mkdir -p train/sunflower

%mkdir -p train/rose

%mkdir -p train/dandelion



%mkdir -p valid/daisy

%mkdir -p valid/tulip

%mkdir -p valid/sunflower

%mkdir -p valid/rose

%mkdir -p valid/dandelion



%cd ..



# You can verify that everything went correctly using ls command
for category in fl_count.index:

    samples = flowers['image'][flowers['category'] == category].values

    perm = np.random.permutation(samples)

    # Copy first 30 samples to the validation directory and rest to the train directory

    for i in range(30):

        name = perm[i].split('/')[-1]

        shutil.copyfile(perm[i],'./data/valid/' + str(category) + '/'+ name)

    for i in range(31,len(perm)):

        name = perm[i].split('/')[-1]

        shutil.copyfile(perm[i],'./data/train/' + str(category) + '/' + name)
# Define the generators



batch_size = 8

# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)



# this is a generator that will read pictures found in

# subfolers of 'data/train', and indefinitely generate

# batches of augmented image data

train_generator = train_datagen.flow_from_directory(

        'data/train',  # this is the target directory

        target_size=(150, 150),  # all images will be resized to 150x150

        batch_size=batch_size,

        class_mode='categorical')  # more than two classes



# this is a similar generator, for validation data

validation_generator = test_datagen.flow_from_directory(

        'data/valid',

        target_size=(150,150),

        batch_size=batch_size,

        class_mode='categorical')
def get_model():

    # Get base model 

    base_model = VGG16(include_top=False, input_shape=(150,150,3))

    # Freeze the layers in base model

    for layer in base_model.layers:

        layer.trainable = False

    # Get base model output 

    base_model_ouput = base_model.output

    

    # Add new layers

    x = Flatten()(base_model.output)

    x = Dense(500, activation='relu', name='fc1')(x)

    x = Dropout(0.5)(x)

    x = Dense(5, activation='softmax', name='fc2')(x)

    

    model = Model(inputs=base_model.input, outputs=x)

    return model
# Get the model

model = get_model()

# Compile it

opt = Adam(lr=1e-3, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Summary

model.summary()
# Fit the genertor 

model.fit_generator(

        train_generator,

        steps_per_epoch=4168 // batch_size,

        epochs=50,

        validation_data=validation_generator,

        validation_steps=150 // batch_size)