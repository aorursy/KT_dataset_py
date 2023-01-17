# import the right packages

"""

os package is used to read files and directory structure



"""

from __future__ import absolute_import, division, print_function, unicode_literals



#for contructing the model

import tensorflow as tf

from tensorflow import keras



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



#data manipulation/viz

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import glob



from sklearn.model_selection import train_test_split # for splitting the data into train/val sets

from sklearn.utils import shuffle

import shutil # for copy and moving files





RND = 1993
# list avail files and directories

os.listdir('/kaggle/input/histopathologic-cancer-detection/')
# coutning the num of samples

print("num test: ", len(os.listdir('/kaggle/input/histopathologic-cancer-detection/test')))

print("total training set: ", len(os.listdir('/kaggle/input/histopathologic-cancer-detection/train')))
# Load Data

# train_labels.csv contains list of all image ids and corresponding label

labels = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')

labels.head()
# see how many samples of each label there are

labels['label'].value_counts()
# # uncomment this section if you want to balance the data



# # make the num of neg cases equal to num of pos cases





# SAMPLE_SIZE = 89117



# # take a random sample of class 0 with size equal to num samples in class 1

# df_0 = labels[labels['label'] == 0].sample(SAMPLE_SIZE, random_state = RND)



# # filter out class 1

# df_1 = labels[labels['label'] == 1].sample(SAMPLE_SIZE, random_state = RND)



# # concat the dataframes

# labels_equal = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)



# # shuffle

# labels_equal = shuffle(labels_equal) # shuffle is from sklearn



# #verify the pos and neg cases equal

# labels_equal['label'].value_counts()
# split into train and val sets

df_train, df_val = train_test_split(labels, test_size=57458, 

                                    random_state=RND, stratify = labels['label'])



# we want the validation size to be the same as the test size

# if you want to use the balanced set use "labels_equal"

# by default test size is 0.25



# stratification means that the train_test_split method returns training and test subsets that

# have the same proportions of class labels as the input dataset.
# num of training samples

df_train['label'].value_counts()
# num of val samples

df_val['label'].value_counts()
# create folders to store data



PATH = '/kaggle/'



train_dir = os.path.join(PATH, 'train_set')

os.mkdir(train_dir)



validation_dir = os.path.join(PATH, 'validation_set')

os.mkdir(validation_dir)

#test_dir = os.path.join(PATH, 'test_set')





train_pos_dir = os.path.join(train_dir, 'pos')  # directory with our training cancer positive pictures

os.mkdir(train_pos_dir)



train_neg_dir = os.path.join(train_dir, 'neg')  # directory with our training cancer negative pictures

os.mkdir(train_neg_dir)





validation_pos_dir = os.path.join(validation_dir, 'pos')  

os.mkdir(validation_pos_dir)



validation_neg_dir = os.path.join(validation_dir, 'neg') 

os.mkdir(validation_neg_dir)







# train_pos_dir = os.path.join(train_dir, 'pos')  

# train_neg_dir = os.path.join(train_dir, 'neg')  
"""

directory structure that we are trying to develop:



kaggle/

|__ train_set

    |______ pos: [cat.0.jpg, cat.1.jpg, cat.2.jpg ....]

    |______ neg: [dog.0.jpg, dog.1.jpg, dog.2.jpg ...]

|__ validation_set

    |______ pos: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg ....]

    |______ neg: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg ...]



|__ input

    |______ Histo....

            |______ test

            |______ train

            

            .

            .

            .

            

from: https://www.tensorflow.org/tutorials/images/classification

"""

# Set the id to be the index in labels_equal dataset

labels.set_index('id', inplace=True)



# Get a list of train and val images

train_list = list(df_train['id'])

val_list = list(df_val['id'])
# Transfer the train images into the appropriate folders created



#looping through id's

for image in train_list:

    

    # the id in the csv file does not have the .tif extension therefore we add it here

    fname = image + '.tif'

    

    # get the label for a certain image

    target = labels.loc[image,'label']

    

    # these must match the folder names

    if target == 0:

        label = 'neg'

    if target == 1:

        label = 'pos'

    

    # source path to image

    src = os.path.join('/kaggle/input/histopathologic-cancer-detection/train', fname)

    

    # destination path to image

    dst = os.path.join(train_dir, label, fname)

    

    # copy the image from the source to the destination

    shutil.copyfile(src, dst)
# Transfer the validation images into the appropriate folders created





#looping through id's

for image in val_list:

    

    # the id in the csv file does not have the .tif extension therefore we add it here

    fname = image + '.tif'

    

    # get the label for a certain image

    target = labels.loc[image,'label']

    

    # these must match the folder names

    if target == 0:

        label = 'neg'

    if target == 1:

        label = 'pos'

    

    # source path to image

    src = os.path.join('/kaggle/input/histopathologic-cancer-detection/train', fname)

    

    # destination path to image

    dst = os.path.join(validation_dir, label, fname)

    

    # copy the image from the source to the destination

    shutil.copyfile(src, dst)
# explore the data





num_pos_tr = len(os.listdir(train_pos_dir))

num_neg_tr = len(os.listdir(train_neg_dir))



num_pos_val = len(os.listdir(validation_pos_dir))

num_neg_val = len(os.listdir(validation_neg_dir))



total_train = num_pos_tr + num_neg_tr

total_val = num_pos_val + num_neg_val



print('total training cancer positive images:', num_pos_tr)

print('total training cancer negative images:', num_neg_tr)



print('total validation cancer positive images:', num_pos_val)

print('total validation cancer negative images:', num_neg_val)

print("--")

print("Total training images:", total_train)

print("Total validation images:", total_val)
# For convenience, set up variables to use while pre-processing the dataset and training the network.



batch_size = 128

epochs = 15

IMG_HEIGHT = 96

IMG_WIDTH = 96

IMG_DEPTH = 3
# Data Preperation



"""

Format the images into appropriately pre-processed floating point tensors before feeding to the network:



-Read images from the disk.

-Decode contents of these images and convert it into proper grid format as per their RGB content.

-Convert them into floating point tensors.

-Rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer 

to deal with small input values.

"""
# ImageDataGenerator class provided by tf.keras: It can read images from disk and preprocess them into 

# proper tensors. It will also set up generators that convert these images into batches of tensors—helpful 

# when training the network.



# Using this because we do not have enough memory to save *all* training images together to feed to network

#Generator for our training data

train_image_generator = ImageDataGenerator(rescale=1./255,

                                          rotation_range=45,

                                          zoom_range=0.5) # applied zoom and rotation augmentations

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,

                                                           directory=train_dir,

                                                           shuffle=True,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           color_mode = 'rgb',

                                                           class_mode='binary')



# color mode is rgb by defualt
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,

                                                              directory=validation_dir,

                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                              class_mode='binary')


sample_training_images, sample_training_labels = next(train_data_gen)



"""

The next function returns a batch from the dataset. The return value of next 

function is in form of (x_train, y_train) where x_train is training features and 

y_train, its labels. Discard the labels to only visualize the training images.

"""
# This function will plot images in the form of a grid with 1 row and 5 columns where 

# images are placed in each column.

def plotImages(images_arr, label):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()

    

plotImages(sample_training_images[:5], sample_training_labels[:5])



print(sample_training_labels[:5])
"""

The model consists of three convolution blocks with a max pool layer in each of them. 

There's a fully connected layer with 512 units on top of it that is activated by a relu 

activation function.

"""

# # original

# model = Sequential([

#     Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

#     MaxPooling2D(),

#     Conv2D(32, 3, padding='same', activation='relu'),

#     MaxPooling2D(),

#     Conv2D(64, 3, padding='same', activation='relu'),

#     MaxPooling2D(),

#     Flatten(),

#     Dense(512, activation='relu'),

#     Dense(1)

# ])



#dropout



model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Dropout(0.2),

    

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.2),

    

    Flatten(),

    Dense(512, activation='relu'),

    Dropout(0.2),



    Dense(1, activation = "sigmoid")

])
# Compile the model

# additional reference: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data



# METRICS = [

#       keras.metrics.TruePositives(name='tp'),

#       keras.metrics.FalsePositives(name='fp'),

#       keras.metrics.TrueNegatives(name='tn'),

#       keras.metrics.FalseNegatives(name='fn'), 

#       keras.metrics.BinaryAccuracy(name='accuracy'),

#       keras.metrics.Precision(name='precision'),

#       keras.metrics.Recall(name='recall'),

#       keras.metrics.AUC(name='auc'),

# ]



model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))





# using AUC as our metric since the competition will be judged using that

# for imbalanced data accuracy is not a good metric





# Model Summary

# View all the layers



model.summary()
# Train the model

    

history = model.fit(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=total_val // batch_size

)
# Visualize the training results



loss=history.history['loss']

val_loss=history.history['val_loss']



epochs_range = range(epochs)



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
# # Put the predictions into a dataframe.

# # The columns need to be oredered to match the output of the previous cell

# predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)



# df_preds = pd.DataFrame(predictions, columns=['no_tumor_tissue', 'has_tumor_tissue'])



# # Get the true labels

# y_true = test_gen.classes



# # Get the predicted labels as probabilities

# y_pred = df_preds['has_tumor_tissue']





# from sklearn.metrics import roc_auc_score



# roc_auc_score(y_true, y_pred)
#reminder of how stratify works



# X = np.arange(10).reshape((5, 2))



# X[0,0] = 0

# X[1,0] = 0

# X[2,0] = 1

# X[3,0] = 1

# X[4,0] = 0



# X_train,  y_train = train_test_split(

#     X, test_size=0.33, random_state=42, stratify = X[:,0])



# X_train

# y_train