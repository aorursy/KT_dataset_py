# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import json

import seaborn as sns



from tensorflow import keras

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras.models import Sequential

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import Adam, RMSprop



import matplotlib.pyplot as plt

%matplotlib inline
train_images_dir = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/'

test_images_dir = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/test/'



train_metadata_file_path = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/metadata.json'

test_metadata_file_path = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/test/metadata.json'



num_classes = 32093 + 1

batch_size = 16



steps_per_epoch = int(num_classes / batch_size)



img_height = 1000

img_width = 661



epochs_num = 5
with open(train_metadata_file_path, 'r', encoding='utf-8', errors='ignore') as f:

    train_metadata_json = json.load(f)
#Let's see presented keys

train_metadata_json.keys()
#Create Pandas DataFrame per each data type

train_metadata = pd.DataFrame(train_metadata_json['annotations'])



train_categories = pd.DataFrame(train_metadata_json['categories'])

train_categories.columns = ['family', 'genus', 'category_id', 'category_name']



train_images = pd.DataFrame(train_metadata_json['images'])

train_images.columns = ['file_name', 'height', 'image_id', 'license', 'width']



train_regions = pd.DataFrame(train_metadata_json['regions'])

train_regions.columns = ['region_id', 'region_name']



#Combine DataFrames

train_data = train_metadata.merge(train_categories, on='category_id', how='outer')

train_data = train_data.merge(train_images, on='image_id', how='outer')

train_data = train_data.merge(train_regions, on='region_id', how='outer')



#Remove NaN values

train_data = train_data.dropna()



# Update data types

train_data = train_data.astype({'category_id': 'int32',

                                'id': 'int32',

                                'image_id': 'int32',

                                'region_id': 'int32',

                                'height': 'int32',

                                'license': 'int32',

                                'width': 'int32'})



train_data.info()



#Save DataFrame for future usage.

train_data.to_csv('train_data.csv', index=False)
del train_categories

del train_images

del train_regions
train_data.head()
with open(test_metadata_file_path, 'r', encoding='utf-8', errors='ignore') as f:

    test_metadata_json = json.load(f)
test_metadata_json.keys()
test_data = pd.DataFrame(test_metadata_json['images'])



test_data = test_data.astype({'height': 'int32',

                              'id': 'int32',

                              'license': 'int32',

                              'width': 'int32'})



test_data.to_csv('test_data.csv', index=False)
datagen_without_augmentation = ImageDataGenerator(rescale=1./255)

datagen_with_augmentation = ImageDataGenerator(rescale=1./255, 

                                               featurewise_center=False,

                                               samplewise_center=False,

                                               featurewise_std_normalization=False,

                                               samplewise_std_normalization=False,

                                               zca_whitening=False,

                                               rotation_range = 10,

                                               zoom_range = 0.1,

                                               width_shift_range=0.1,

                                               height_shift_range=0.1,

                                               horizontal_flip=True,

                                               vertical_flip=False)



train_datagen = datagen_with_augmentation.flow_from_dataframe(dataframe=train_data, 

                                                                 directory=train_images_dir, 

                                                                 x_col='file_name', 

                                                                 y_col='category_id',

                                                                 class_mode="raw",

                                                                 batch_size=batch_size,

                                                                 color_mode = 'rgb',

                                                                 target_size=(img_height,img_width)

                                                             )



val_datagen = datagen_without_augmentation.flow_from_dataframe(dataframe=train_data, 

                                                                 directory=train_images_dir, 

                                                                 x_col='file_name', 

                                                                 y_col='category_id',

                                                                 class_mode="raw",

                                                                 batch_size=batch_size,

                                                                 color_mode = 'rgb',

                                                                 target_size=(img_height,img_width))



#test_datagen = datagen_without_augmentation.flow_from_dataframe(dataframe=test_data,

#                                                               directory=test_images_dir,

#                                                               x_col='file_name',

#                                                               color_mode = 'rgb',

#                                                               class_mode=None,

#                                                               target_size=(img_height,img_width))
def generator_wrapper(generator, num_of_classes):

    for (X_vals, y_vals) in generator:

        Y_categorical = to_categorical(y_vals, num_classes=num_of_classes)

        

        yield (X_vals, Y_categorical)        

        

train_datagen_wrapper = generator_wrapper(train_datagen, num_classes)

val_datagen_wrapper = generator_wrapper(val_datagen, num_classes)
model = Sequential()



model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=(img_height, img_width, 3), padding='Same', strides=2))

model.add(Conv2D(64, kernel_size=5, activation='relu', padding='Same', strides=2))

model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, kernel_size=3, activation='relu', padding='Same', strides=2))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=3, activation='relu', padding='Same', strides=2))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(num_classes / 100))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))



optimizer = RMSprop(lr=0.001)



model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
import time



start = time.time()



# history = model.fit_generator(train_datagen_wrapper, 

#                               epochs=epochs_num, 

#                               validation_data=val_datagen_wrapper, 

#                               steps_per_epoch=steps_per_epoch, 

#                               validation_steps=steps_per_epoch)

# 



end = time.time()



print(f"\nLearning took {end - start}")