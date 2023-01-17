from numpy.random import seed

seed(101)

from tensorflow import set_random_seed

set_random_seed(101)



import pandas as pd

import numpy as np





import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.optimizers import Adam



import os

import cv2



from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import itertools

import shutil

import matplotlib.pyplot as plt

%matplotlib inline


IMAGE_SIZE = 96

IMAGE_CHANNELS = 3



SAMPLE_SIZE = 80000 # the number of images we use from each of the two classes

os.listdir('../input')


print(len(os.listdir('../input/train')))

print(len(os.listdir('../input/test')))
df_data = pd.read_csv('../input/train_labels.csv')

df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']

print(df_data.shape)
df_data['label'].value_counts()
df_data.head()
df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)

df_1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)

# concat the dataframes

df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)

df_data = shuffle(df_data)

df_data['label'].value_counts()
df_data.head()
y = df_data['label']

df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)

print(df_train.shape)

print(df_val.shape)
df_train['label'].value_counts()
df_val['label'].value_counts()
# Create a new directory

base_dir = 'base_dir'

os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train_dir')

os.mkdir(train_dir)

# val_dir

val_dir = os.path.join(base_dir, 'val_dir')

os.mkdir(val_dir)

# create new folders inside train_dir

no_tumor_tissue = os.path.join(train_dir, 'a_no_tumor_tissue')

os.mkdir(no_tumor_tissue)

has_tumor_tissue = os.path.join(train_dir, 'b_has_tumor_tissue')

os.mkdir(has_tumor_tissue)

# create new folders inside val_dir

no_tumor_tissue = os.path.join(val_dir, 'a_no_tumor_tissue')

os.mkdir(no_tumor_tissue)

has_tumor_tissue = os.path.join(val_dir, 'b_has_tumor_tissue')

os.mkdir(has_tumor_tissue)



df_data.set_index('id', inplace=True)




train_list = list(df_train['id'])

val_list = list(df_val['id'])



# Transfer the train images



for image in train_list:

    

    # adding extension .tif to image name

    fname = image + '.tif'

    # label for a certain image

    target = df_data.loc[image,'label']

    

    if target == 0:

        label = 'a_no_tumor_tissue'

    if target == 1:

        label = 'b_has_tumor_tissue'

    

    # source path to image

    src = os.path.join('../input/train', fname)

    # destination path to image

    dst = os.path.join(train_dir, label, fname)

    # copy the image from the source to the destination

    shutil.copyfile(src, dst)





# Transfer the val images



for image in val_list:

    

    # the id in the csv file does not have the .tif extension therefore we add it here

    fname = image + '.tif'

    # get the label for a certain image

    target = df_data.loc[image,'label']

    

    # these must match the folder names

    if target == 0:

        label = 'a_no_tumor_tissue'

    if target == 1:

        label = 'b_has_tumor_tissue'

    



    # source path to image

    src = os.path.join('../input/train', fname)

    # destination path to image

    dst = os.path.join(val_dir, label, fname)

    # copy the image from the source to the destination

    shutil.copyfile(src, dst)

    





   
# check how many train images we have in each folder



print(len(os.listdir('base_dir/train_dir/a_no_tumor_tissue')))

print(len(os.listdir('base_dir/train_dir/b_has_tumor_tissue')))

# check how many val images we have in each folder



print(len(os.listdir('base_dir/val_dir/a_no_tumor_tissue')))

print(len(os.listdir('base_dir/val_dir/b_has_tumor_tissue')))

train_path = 'base_dir/train_dir'

valid_path = 'base_dir/val_dir'

test_path = '../input/test'



num_train_samples = len(df_train)

num_val_samples = len(df_val)

train_batch_size = 10

val_batch_size = 10





train_steps = np.ceil(num_train_samples / train_batch_size)

val_steps = np.ceil(num_val_samples / val_batch_size)
datagen = ImageDataGenerator(rescale=1.0/255)



train_gen = datagen.flow_from_directory(train_path,

                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                        batch_size=train_batch_size,

                                        class_mode='categorical')



val_gen = datagen.flow_from_directory(valid_path,

                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                        batch_size=val_batch_size,

                                        class_mode='categorical')



# Note: shuffle=False causes the test dataset to not be shuffled

test_gen = datagen.flow_from_directory(valid_path,

                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),

                                        batch_size=1,

                                        class_mode='categorical',

                                        shuffle=False)




kernel_size = (3,3)

pool_size= (2,2)

first_filters = 32

second_filters = 64

third_filters = 128



dropout_conv = 0.3

dropout_dense = 0.3





model = Sequential()

model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))

model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))

model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))

model.add(MaxPooling2D(pool_size = pool_size)) 

model.add(Dropout(dropout_conv))



model.add(Conv2D(second_filters, kernel_size, activation ='relu'))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))

model.add(MaxPooling2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))

model.add(MaxPooling2D(pool_size = pool_size))

model.add(Dropout(dropout_conv))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(dropout_dense))

model.add(Dense(2, activation = "softmax"))



model.summary()

model.compile(Adam(lr=0.0001), loss='binary_crossentropy', 

              metrics=['accuracy'])
filepath = "model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 

                             save_best_only=True, mode='max')



reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 

                                   verbose=1, mode='max', min_lr=0.00001)

                              

                              

callbacks_list = [checkpoint, reduce_lr]



history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 

                    validation_data=val_gen,

                    validation_steps=val_steps,

                    epochs=5, verbose=1,

                   callbacks=callbacks_list)
shutil.rmtree('base_dir')

shutil.rmtree('test_dir')