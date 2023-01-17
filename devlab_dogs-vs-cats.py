# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile



zip = zipfile.ZipFile('../input/dogs-vs-cats/train.zip')

zip.extractall('/kaggle/temp')

testzip = zipfile.ZipFile('../input/dogs-vs-cats/test1.zip')

testzip.extractall('/kaggle/temp')

len(os.listdir('/kaggle/temp/train')), len(os.listdir('/kaggle/temp/test1'))
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow.keras.preprocessing.image as image

import matplotlib.pyplot as plt

import tensorflow as tf
train_dir = '/kaggle/temp/train'

test_dir = '/kaggle/temp/test1'



train_dogs_dir = os.path.join(train_dir, 'dogs')

train_cats_dir = os.path.join(train_dir, 'cats')

if not os.path.exists(train_dogs_dir):

    os.mkdir(train_dogs_dir)

if not os.path.exists(train_cats_dir):

    os.mkdir(train_cats_dir)



filenames = [name for name in os.listdir(train_dir) if name.endswith('.jpg')]

for filename in filenames:

    category = 'dogs' if filename.split('.')[0] == 'dog' else 'cats'

    os.rename(os.path.join(train_dir, filename), os.path.join(train_dir, category, filename))
my_model = Sequential()

my_model.add(Xception(include_top=False, pooling='avg'))

# my_model.add(ResNet50(include_top=False, pooling='avg'))

my_model.add(Dense(2, activation='softmax'))

my_model.layers[0].trainable = False
my_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)



train_generator = data_generator.flow_from_directory(

        train_dir,

        subset='training',

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        train_dir,

        subset='validation',

        class_mode='categorical')



my_model.fit_generator(train_generator, steps_per_epoch=3, epochs=7, validation_data=validation_generator, validation_steps=1)
filenames = os.listdir(test_dir)



df = pd.DataFrame({'filename': filenames})



test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



test_generator = test_datagen.flow_from_dataframe(df, directory=test_dir, x_col='filename', y_col=None, class_mode=None, shuffle=False)



x = test_generator.next()



# plt.figure(figsize=(20, 10))

# for i in range(0, 10):

#     plt.subplot(2, 5, i+1)

#     plt.imshow(x[i])

# plt.show()



preds = my_model.predict_generator(test_generator)



score = np.argmax(preds, axis=1)



id = df['filename'].str.split('.').str[0]

df['id'] = id.astype('int')

df['label'] = score



df = df.drop(['filename'], axis=1)

df = df.sort_values('id')



df.to_csv('/kaggle/working/submission.csv', index=False)