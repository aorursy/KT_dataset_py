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
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import tensorflow.keras as keras

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2

print("matplotlib version: {}".format(matplotlib.__version__))

print("numpy version: {}".format(np.__version__))

print("pandas version: {}".format(pd.__version__))

print("matplotlib backend: {}".format(matplotlib.get_backend()))

print("Tensorflow version: {}".format(tf.__version__))

print("tf.keras version: {}".format(tf.keras.__version__))
train_csv_df = pd.read_csv("/kaggle/input/bengaliai-cv19/train.csv")

train_csv_df.head()
train_csv_df.describe()
train_image_data_df_0 = pd.read_parquet("/kaggle/input/bengaliai-cv19/train_image_data_0.parquet")

train_image_data_df_0.head()
print("Total number of training images(rows) in each training parquet file: {}".format(train_image_data_df_0.shape[0]))

print("Total number of pixels in each image(colums) of the training dataset: {}".format(train_image_data_df_0.shape[1]))

print("Total number of training images: {}".format(train_csv_df['image_id'].count()))
from itertools import cycle, islice



top_n_grapheme_root_df = train_csv_df['grapheme_root'].value_counts().nlargest(15)

color_list = list(islice(cycle(['b', 'r', 'g', 'c', 'y']), None, len(top_n_grapheme_root_df)))

ax = top_n_grapheme_root_df.plot(kind = 'barh', color = color_list, rot = 0, figsize = (15, 10), fontsize = 15)

ax.set_xlabel(xlabel = 'Image Count', fontsize = 20)

ax.set_ylabel(ylabel = 'Grapheme Root', fontsize = 20)

ax.set_title(label = 'Grapheme Root Frequency Plot (TopN)', fontsize = 25)
vowel_diacritic_df = train_csv_df['vowel_diacritic'].value_counts()





color_list = list(islice(cycle(['b', 'r', 'g', 'c', 'y']), None, len(vowel_diacritic_df)))

ax = vowel_diacritic_df.plot(kind = 'barh', color = color_list, rot = 0, figsize = (15, 10), fontsize = 15)

ax.set_xlabel(xlabel = 'Image Count', fontsize = 20)

ax.set_ylabel(ylabel = 'Vowel Diacritic', fontsize = 20)

ax.set_title(label = 'Vowel Diacritic Frequency Plot', fontsize = 25)
consonant_diacritic_df = train_csv_df['consonant_diacritic'].value_counts()





color_list = list(islice(cycle(['b', 'r', 'g', 'c', 'y']), None, len(consonant_diacritic_df)))

ax = consonant_diacritic_df.plot(kind = 'barh', color = color_list, rot = 0, figsize = (15, 10), fontsize = 15)

ax.set_xlabel(xlabel = 'Image Count', fontsize = 20)

ax.set_ylabel(ylabel = 'Consonant Diacritic', fontsize = 20)

ax.set_title(label = 'Consonant Diacritic Frequency Plot', fontsize = 25)
test_csv_df = pd.read_csv("/kaggle/input/bengaliai-cv19/test.csv")

submission_df = pd.read_csv("/kaggle/input/bengaliai-cv19/sample_submission.csv")

class_map_df = pd.read_csv("/kaggle/input/bengaliai-cv19/class_map.csv")
test_csv_df.head()
print("Total number of rows in test_csv dataframe: {}".format(train_csv_df['image_id'].count()))
submission_df.head()
class_map_df.head()
class_map_df.groupby(['component_type'])['label'].max()
# one hot encoding of labels



one_hot_df = pd.concat([

    train_csv_df[["image_id"]],

    pd.get_dummies(train_csv_df.grapheme_root, prefix="grapheme_root"),

    pd.get_dummies(train_csv_df.vowel_diacritic, prefix="vowel_diacritic"),

    pd.get_dummies(train_csv_df.consonant_diacritic, prefix="consonant_diacritic"),

], axis = 1)



one_hot_df.head()
train_data_df = train_image_data_df_0.set_index('image_id').join(one_hot_df.set_index('image_id'))

train_data_df.head()
# Label columns per attribute type

_consonant_diacritic_cols_ = [col for col in train_data_df.columns if col.startswith("consonant_diacritic")]

_grapheme_root_cols_ = [col for col in train_data_df.columns if col.startswith("grapheme_root")]

_vowel_diacritic_cols_ = [col for col in train_data_df.columns if col.startswith("vowel_diacritic")]





class BegaliImageDataGenerator(keras.utils.Sequence):



  def __init__(self, df, augmentation = None, policy = None, new_shape = (128,128) , batch_size=32, shuffle=True):

        self.df = df.iloc[:, 0:137*236]

        self.label_df = df.iloc[:, 137*236: ]

        self.batch_size=batch_size

        self.shuffle = shuffle

        self.augment = augmentation

        self.policy = policy

        self.on_epoch_end()



  def __len__(self):

        return int(np.floor(self.df.shape[0] / self.batch_size))



  def __getitem__(self, index):

        """fetch batched images and targets"""

        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)

        items = self.df.iloc[batch_slice]

        label_items = self.label_df.iloc[batch_slice]

        if not self.augment:

          image = np.stack([cv2.resize(np.reshape(item.values, (137, 236)).astype(np.float32), new_shape) for _, item in items.iterrows()])

        else:

          image = np.stack([cv2.resize(np.reshape(item.values, (137, 236)).astype(np.float32), new_shape) for _, item in items.iterrows()])



        target = {

            "consonant_diacritic_output": label_items[_consonant_diacritic_cols_].values,

            "grapheme_root_output": label_items[_grapheme_root_cols_].values,

            "vowel_diacritic_output": label_items[_vowel_diacritic_cols_].values,

        }

        

        image = image.reshape(image.shape[0], image.shape[1], image.shape[2], 1)

        return image, target



  def on_epoch_end(self):

        """Updates indexes after each epoch"""

        if self.shuffle == True:

            self.df = self.df.sample(frac=1).reset_index(drop=True)
# Show train images from ImageDataGenerator after resizing to 128x128 size



new_shape = (128,128)

train_generator32 = BegaliImageDataGenerator(train_data_df, policy = None, batch_size=32, new_shape = new_shape, augmentation = ImageDataGenerator(

        horizontal_flip=False,

        vertical_flip=False,

    ))



x, y = next(iter(train_generator32))

print(x.shape)



plt.figure(figsize=(32, 16))

for i, (img, label) in enumerate(zip(x, y['consonant_diacritic_output'])):

    plt.subplot(4, 8, i+1)

    plt.axis('off')

    plt.imshow(img.reshape(img.shape[0],img.shape[1]), interpolation="nearest", cmap = 'gray')