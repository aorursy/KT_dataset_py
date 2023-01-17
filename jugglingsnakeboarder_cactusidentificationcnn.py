import zipfile



# unzip the files so that you can see them..

with zipfile.ZipFile("/kaggle/input/aerial-cactus-identification/train.zip","r") as z:

    z.extractall("/kaggle/working/train")

with zipfile.ZipFile("/kaggle/input/aerial-cactus-identification/test.zip","r") as z:

    z.extractall("/kaggle/working/test")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import Image

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras import regularizers

from keras import layers,models

from keras.layers.core import Dense

from keras.layers import Conv2D, MaxPool2D, Flatten



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import cv2

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_directory = "/kaggle/working/train/train"

test_directory = "/kaggle/working/test/"
train_df = pd.read_csv('../input/aerial-cactus-identification/train.csv',dtype=str) # "dtype=str" is importend for the later flow_from_dataframe-method

train_df
test_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv',dtype=str) # "dtype=str" is importend for the later flow_from_dataframe-method

test_df
#have a look at the first image in train-directory

import cv2

Image(os.path.join("/kaggle/working/train/train",train_df.iloc[0,0]),width=32,height=32)
# using the ImageDataGenerator-class from keras for preparing the data

main_datagenerator=ImageDataGenerator(rescale=1./255)

# store all image-data from the train directory in an ImageDataGenerator

# the data-generator produces packages of 150x(images,label) with shape (150,32,32,3)

# the 'has_cactus'-entries come from the train_df

# the images come from the train_directory

# splitting them into train- and validation-data

train_data_batch_size=150

train_datagenerator = main_datagenerator.flow_from_dataframe(dataframe=train_df[:15001],directory=train_directory,x_col="id",y_col="has_cactus",class_mode='binary',target_size=(32,32),batch_size=train_data_batch_size)

val_data_batch_size=20

val_datagenerator = main_datagenerator.flow_from_dataframe(dataframe=train_df[15000:],directory=train_directory,x_col="id",y_col="has_cactus",class_mode='binary',target_size=(32,32),batch_size=val_data_batch_size)
# take a look at the generator-outputs

for data, labels in train_datagenerator:

    print("data-shape: ",data.shape)

    print("label-shape: ",labels.shape)

    break # dont want to see the hole generating shapes
model=models.Sequential()

# first the CNN

model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(32,32,3)))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(32,32,3)))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3),padding='same',activation='relu',input_shape=(32,32,3)))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3),padding='same',activation='relu',input_shape=(32,32,3)))

model.add(MaxPool2D((2,2)))

# following by an MLP

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dense(128,activation='relu'))

model.add(Dense(1,activation='sigmoid'))



#model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l2(0.001),activation='relu'))

#model.add(Dense(1, activation='sigmoid'))















model.summary()
model.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop(lr=1e-4),metrics=['acc'])
number_of_epochs = 50 #50

steps = 100          #100



history=model.fit_generator(

    train_datagenerator,

    steps_per_epoch=steps,

    epochs=number_of_epochs,

    validation_data=val_datagenerator,

    validation_steps=20,

    verbose=1)

# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# creating a ImageDataGenerator with one-single-image-per-batch an no shuffle



test_generator = main_datagenerator.flow_from_directory(

    directory=test_directory,

    target_size=(32,32),

    batch_size=1,# no packages

    class_mode='binary',

    shuffle=False # maintain the sequence

)
prediction=model.predict_generator(test_generator,verbose=1)

pred_binary = [0 if value<0.50 else 1 for value in prediction] 

pred_binary = np.array(pred_binary)

pred_binary.reshape(4000,1)

print(pred_binary)

import collections

collections.Counter(pred_binary)
# get the test-images as a list of '*.jpg-name'-strings from the test directory

test_files = test_df['id'] # os.listdir("/kaggle/working/test/test")

test_files

# constructing the dataframe (2 columns with entries 'id *.jpg-name' '0/1 has cactus')

sub_file = pd.DataFrame(data = {'id': test_files, 'has_cactus': pred_binary.reshape(-1).tolist()})

sub_file
import shutil

shutil.rmtree('/kaggle/working/test')

shutil.rmtree('/kaggle/working/train')

#produce the submission file

sub_file.to_csv('submission.csv', index=False)