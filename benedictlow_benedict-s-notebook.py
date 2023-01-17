import pandas as pd

from pathlib import Path

from collections import defaultdict

import matplotlib.pyplot as plt

import seaborn as sns

import os, sys



train_data = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

train_data.head()
# List if imageId in the Train Folder which have Defects

imageId = pd.DataFrame(train_data['ImageId'])

imageId.head()
imageId.info()
train_data["ClassId"].value_counts(ascending=True, dropna = False)
defect_visual = train_data["ClassId"].value_counts()

plt.figure(figsize=(7,4))

sns.barplot(defect_visual.index, defect_visual.values, alpha = 0.8)

plt.title("Number of Steel Defect in Train Dataset")

plt.xlabel("Defect Class")

plt.ylabel("Defect Count")
images_path = Path("../input/severstal-steel-defect-detection/train_images")

images = [f for f in os.listdir(images_path)]



# No. of Images in Training Folder

len(images)
# Create Validation Set from Training Set

from sklearn.model_selection import train_test_split



train, val = train_test_split(images, train_size=0.8)
# No. of Images Used for Training

len(train)
fileList = os.listdir('../input/severstal-steel-defect-detection')

for f in fileList:

    print(f)
# unpack zip files in path directory

for i in fileList:

    if "zip" in i:

        unpack_archive(path + i, path, 'zip')
fileList = os.listdir('../input/severstal-steel-defect-detection') # get updated file list in path directory

# list files/folders

for f in fileList:

    print(f)
shutil.os.mkdir('../input/severstal-steel-defect-detection/' + 'valid')
# Create Validation Folder

import shutil

from shutil import unpack_archive

from subprocess import check_output



val_folder = shutil.os.mkdir('../input/severstal-steel-defect-detection/val_images')



os.path.join('val_images') 

os.makedirs(val_folder)

os.listdir()
# Move Validation Images to Validation Folder

import shutil



for i in images:

  if (i not in train):

    old_path = "../input/severstal-steel-defect-detection/train_images/" + i

    new_path = 'val_folder' + i

    shutil.move(old_path, new_path)
# Create Folders to Seperate Images With and Without Defects

os.makedirs(train_folder + '/y')

os.makedirs(train_folder + '/n')

os.makedirs(val_folder + '/y')

os.makedirs(val_folder + '/n')
# Extract imageId with Defects into an Array

for index, row in imageId.iteritems():

  values = row.values



values
# Categorize Images in Training Folder

for i in train:

  old_path = 'train_images/' + i

  if (i in values):

    new_path = 'train_images/y/' + i

  else: 

    new_path = 'train_images/n/' + i

  shutil.move(old_path, new_path)
# Categorize Images in Validation Folder

for i in val:

  old_path = 'val_images/' + i

  if (i in values):

    new_path = 'val_images/y/' + i

  else: 

    new_path = 'val_images/n/' + i

  shutil.move(old_path, new_path)
# List Folders and Number of Files (Training)

print("Directory, Number of Files")

for root, subdirs, files in os.walk("train_images"):

    print(root, len(files))
# List Folders and Number of Files (Validation)

print("Directory, Number of Files")

for root, subdirs, files in os.walk("val_images"):

    print(root, len(files))
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# All Images will be Rescaled by 1./255. We Apply Data Augmentation Here.

train_datagen = ImageDataGenerator(rotation_range=40,

                                   width_shift_range=0.1,

                                   height_shift_range=0.1,

                                   rescale=1./255,

                                   shear_range=0.1,

                                   zoom_range=0.1,

                                   horizontal_flip=True,

                                   vertical_flip=True,

                                   fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)
bs = 16 

img_size = (256, 512)



train_gen = train_datagen.flow_from_directory(

    directory=train_folder,

    target_size=img_size,

    batch_size=bs,

    class_mode='binary'

)



test_gen = test_datagen.flow_from_directory(

    directory=val_folder,

    target_size=img_size,

    batch_size=bs,

    class_mode='binary'

)
from keras.applications import DenseNet121

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Flatten, Dense, Dropout, BatchNormalization



def buildModel1():

  dense_net = DenseNet121(

      include_top=False,

      input_shape=(256, 512, 3), # (width, height, colorchannel)

      weights='imagenet'

  )



  model = Sequential()

  model.add(dense_net)

  model.add(GlobalAveragePooling2D())

  model.add(BatchNormalization())

  model.add(Dropout(0.5))

  model.add(Dense(512, activation='relu'))

  model.add(BatchNormalization())

  model.add(Dropout(0.5))

  model.add(Dense(1, activation='sigmoid'))



  model.compile(

      loss='binary_crossentropy',

      optimizer='adam',

      metrics=['accuracy', f1_m, precision_m, recall_m]

  )



  return model
history1 = buildModel1().fit_generator(

          train_gen, # train generator has 12568 train images but we are not using all of them

          steps_per_epoch=786, # training 12568 images = 786 steps x 16 images per batch

          epochs=25,

          validation_data=test_gen, # validation generator has 5,000 validation images

          validation_steps=158 # validating on 2514 images = 158 steps x 16 images per batch

)
import matplotlib.pyplot as plt



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