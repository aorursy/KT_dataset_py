# ATTENTION: Please do not alter any of the provided code in the exercise. Only add your own code where indicated

# ATTENTION: Please do not add or remove any cells in the exercise. The grader will check specific cells based on the cell position.

# ATTENTION: Please use the provided epoch values when training.



# In this exercise you will train a CNN on the FULL Cats-v-dogs dataset

# This will require you doing a lot of data preprocessing because

# the dataset isn't split into training and validation for you

# This code block has all the required inputs

import os

import zipfile

import random

import tensorflow as tf

import shutil

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile

from os import getcwd
path_cats_and_dogs = f"{getcwd()}/../tmp2/cats-and-dogs.zip"

shutil.rmtree('/tmp')



local_zip = path_cats_and_dogs

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp')

zip_ref.close()

print(len(os.listdir('/tmp/PetImages/Cat/')))

print(len(os.listdir('/tmp/PetImages/Dog/')))



# Expected Output:

# 1500

# 1500
# Use os.mkdir to create your directories

# You will need a directory for cats-v-dogs, and subdirectories for training

# and testing. These in turn will need subdirectories for 'cats' and 'dogs'

req_dir= [

    '/tmp/cats-v-dogs',

    '/tmp/cats-v-dogs/training',

    '/tmp/cats-v-dogs/testing',

    '/tmp/cats-v-dogs/training/cats',

    '/tmp/cats-v-dogs/training/dogs',

    '/tmp/cats-v-dogs/testing/cats',

    '/tmp/cats-v-dogs/testing/dogs',

]

for dirc in req_dir:

    try:

        os.mkdir(dirc)

        print(dirc,'created')

    except OSError:

        pass
# Write a python function called split_data which takes

# a SOURCE directory containing the files

# a TRAINING directory that a portion of the files will be copied to

# a TESTING directory that a portion of the files will be copie to

# a SPLIT SIZE to determine the portion

# The files should also be randomized, so that the training set is a random

# X% of the files, and the test set is the remaining files

# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9

# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir

# and 10% of the images will be copied to the TESTING dir

# Also -- All images should be checked, and if they have a zero file length,

# they will not be copied over

#

# os.listdir(DIRECTORY) gives you a listing of the contents of that directory

# os.path.getsize(PATH) gives you the size of the file

# copyfile(source, destination) copies a file from source to destination

# random.sample(list, len(list)) shuffles a list

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

    all_files=[]

    for file_name in os.listdir(SOURCE):

        file_path = SOURCE + file_name

        

        if os.path.getsize(file_path):

            all_files.append(file_name)

        else:

            pass

    n_files = len(all_files)

    split_point=int(n_files*SPLIT_SIZE)

    shuffled=random.sample(all_files,n_files)

    train_set=shuffled[:split_point]

    test_set=shuffled[split_point:]

    

    for file_name in train_set:

        copyfile(SOURCE+file_name,TRAINING+file_name)

        

    for file_name in test_set:

        copyfile(SOURCE+file_name,TESTING+file_name)

# YOUR CODE STARTS HERE

# YOUR CODE ENDS HERE





CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"

TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"

TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"

DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"

TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"

TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"



split_size = .9

split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)

split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))

print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))

print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))

print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))



# Expected output:

# 1350

# 1350

# 150

# 150
# DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS

# USE AT LEAST 3 CONVOLUTION LAYERS

model = tf.keras.models.Sequential([

# YOUR CODE HERE

    tf.keras.layers.Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])



model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])




TRAINING_DIR = '/tmp/cats-v-dogs/training'

train_datagen = ImageDataGenerator(

    rescale=1 / 255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

train_generator = train_datagen.flow_from_directory(

    TRAINING_DIR,

    batch_size=10,

    class_mode='binary',

    target_size=(150, 150)

)



VALIDATION_DIR = '/tmp/cats-v-dogs/testing'

validation_datagen = ImageDataGenerator(

    rescale=1 / 255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'



)

validation_generator = validation_datagen.flow_from_directory(

    VALIDATION_DIR,

    batch_size=10,

    class_mode='binary',

    target_size=(150, 150)

)

history = model.fit_generator(train_generator,

                              epochs=2,

                              verbose=1,

                              validation_data=validation_generator)

# PLOT LOSS AND ACCURACY

%matplotlib inline



import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")





plt.title('Training and validation loss')



# Desired output. Charts with training and validation metrics. No crash :)
# Now click the 'Submit Assignment' button above.