# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir('../input/v2-plant-seedlings-dataset')
from numpy.random import seed
seed(101)

import pandas as pd
import numpy as np

import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import os
import cv2

import imageio
import skimage
import skimage.io
import skimage.transform

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline
folder_list = os.listdir('../input/v2-plant-seedlings-dataset')
total_images = 0
# loop through each folder
for folder in folder_list:
    # set the path to a folder
    path = '../input/v2-plant-seedlings-dataset/' + str(folder)
    # get a list of images in that folder
    images_list = os.listdir(path)
    # get the length of the list
    num_images = len(images_list)
    
    total_images = total_images + num_images
    # print the result
    print(str(folder) + ':' + ' ' + str(num_images))

# print the total number of images available
print('Total Images: ', total_images)
try:
    all_images_dir = 'images_all'
    os.mkdir(all_images_dir)
except:
    print("Already there")
!ls
folder_list = os.listdir('../input/v2-plant-seedlings-dataset')
for folder in folder_list:
    #path to the folder
    if(folder=='nonsegmentedv2'):
        continue
    path = '../input/v2-plant-seedlings-dataset/' + str(folder)
    #list of all files in the folder
    file_list = os.listdir(path)
    # move the 0 images to images_all
    for fname in file_list:
        # source path to image
        src = os.path.join(path, fname)
        # Change the file name because many images have the same file name.
        # Add the folder name to the existing file name.
        new_fname = str(folder) + '_' + fname
        # destination path to image
        dst = os.path.join(all_images_dir, new_fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
len(os.listdir('images_all'))
image_list = os.listdir('images_all')
#dataframe of all images
df_data = pd.DataFrame(image_list, columns=['image_id'])
df_data.head()
#each filename has the format species_name_565.png
#here we extract the class names for each image
def get_class(x):
    # split into a list
    a = x.split('_')
    # the target is the first index in the list
    cname = a[0]
    return cname
df_data['target'] = df_data['image_id'].apply(get_class)
df_data.head()
df_data.shape
SAMPLE_SIZE=250
IMAGE_SIZE = 128
target_list = os.listdir('../input/v2-plant-seedlings-dataset')
i=target_list.index("nonsegmentedv2")
target_list.pop(i)
#print(target_list)
for target in target_list:
    # Filter out a target and take a random sample
    df = df_data[df_data['target'] == target].sample(SAMPLE_SIZE, random_state=101)
    # if it's the first item in the list
    if target == target_list[0]:
        df_sample = df
    else:
        # Concat the dataframes
        df_sample = pd.concat([df_sample, df], axis=0).reset_index(drop=True)
df_sample['target'].value_counts()
y = df_sample['target']
df_train, df_val = train_test_split(df_sample, test_size=0.10, random_state=101, stratify=y)
print(df_train.shape)
print(df_val.shape)
df_train['target'].value_counts()
df_val['target'].value_counts()
try:
    base_dir = 'base_dir'
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train_dir')
    os.mkdir(train_dir)
    val_dir = os.path.join(base_dir, 'val_dir')
    os.mkdir(val_dir)



    for folder in folder_list:
        folder = os.path.join(train_dir, str(folder))
        os.mkdir(folder)
    for folder in folder_list:
        folder = os.path.join(val_dir, str(folder))
        os.mkdir(folder)
except:
    print("already created")
os.listdir('base_dir/train_dir')
df_data.set_index('image_id', inplace=True)
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])
for image in train_list:
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image
    # get the label for a certain image
    folder = df_data.loc[image,'target']
    # source path to image
    src = os.path.join(all_images_dir, fname)
    # destination path to image
    dst = os.path.join(train_dir, folder, fname)
    
    # resize the image and save it at the new location
    image = cv2.imread(src)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # save the image at the destination
    cv2.imwrite(dst, image)

for image in val_list:
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image
    # get the label for a certain image
    folder = df_data.loc[image,'target']
    # source path to image
    src = os.path.join(all_images_dir, fname)
    # destination path to image
    dst = os.path.join(val_dir, folder, fname)
    # resize the image and save it at the new location
    image = cv2.imread(src)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # save the image at the destination
    cv2.imwrite(dst, image)
folder_list = os.listdir('base_dir/train_dir')
total_images = 0
# loop through each folder
for folder in folder_list:
    # set the path to a folder
    path = 'base_dir/train_dir/' + str(folder)
    # get a list of images in that folder
    images_list = os.listdir(path)
    # get the length of the list
    num_images = len(images_list)
    total_images = total_images + num_images
    # print the result
    print(str(folder) + ':' + ' ' + str(num_images))
# print the total number of images available
print('Total Images: ', total_images)
os.rmdir('base_dir/train_dir/nonsegmentedv2')
folder_list = os.listdir('base_dir/val_dir')
total_images = 0
# loop through each folder
for folder in folder_list:
    # set the path to a folder
    path = 'base_dir/val_dir/' + str(folder)
    # get a list of images in that folder
    images_list = os.listdir(path)
    # get the length of the list
    num_images = len(images_list)
    total_images = total_images + num_images
    # print the result
    print(str(folder) + ':' + ' ' + str(num_images))
# print the total number of images available
print('Total Images: ', total_images)
os.rmdir('base_dir/val_dir/nonsegmentedv2')
train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

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
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', 
                 input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
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
model.add(Dense(12, activation = "softmax"))

model.summary()
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=10, verbose=1,
                    callbacks=callbacks_list)
model.metrics_names
# Print the validation loss and accuracy.

# Here the best epoch will be used.
model.load_weights('model.h5')

val_loss, val_acc = model.evaluate_generator(test_gen, steps=len(df_val))

print('val_loss:', val_loss)
print('val_acc:', val_acc)
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
