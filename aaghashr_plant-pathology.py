import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import h5py

import sys

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

from numpy import save, load

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm

import os

import shutil

from skimage.io import imread

from skimage.transform import resize
train_pd = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

train_pd.shape
test_pd = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

test_pd.shape
train_pd['image_id'] = train_pd['image_id'] + '.jpg'

test_pd['image_id'] = test_pd['image_id'] + '.jpg'
train_pd.head()
test_pd.head()
IMG_PATH = 'images/'



for i in range(5):

    plt.imshow(mpimg.imread(IMG_PATH + train.iloc[i,:]['image_id']))

    if train.iloc[i,:]['healthy'] == 1:

        plt.title('healthy')

    elif train.iloc[i,:]['multiple_diseases'] == 1:

        plt.title('multiple_diseases')

    elif train.iloc[i,:]['rust'] == 1:

        plt.title('rust')

    else:

        plt.title('scab')

    plt.show()
X = train_pd['image_id'].values

y = train_pd.drop(columns=['image_id'])

print('X shape - ', X.shape)

print('y shape - ',y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1, test_size=0.1)
print('X train - ', X_train.shape)

print('y train - ',y_train.shape)

print('X val - ', X_val.shape)

print('y val - ',y_val.shape)
class My_Custom_Generator(keras.utils.Sequence) :

  

  def __init__(self, image_filenames, labels, batch_size) :

    self.image_filenames = image_filenames

    self.labels = labels

    self.batch_size = batch_size

    

    

  def __len__(self) :

    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

  

  

  def __getitem__(self, idx) :

    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]

    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    

    return np.array([

            resize(imread('/kaggle/input/plant-pathology-2020-fgvc7/images/' + str(file_name + '.jpg')), (28, 28, 3))

               for file_name in batch_x])/255.0, np.array(batch_y)
batch_size = 32



my_training_batch_generator = My_Custom_Generator(X_train, y_train, batch_size)

my_validation_batch_generator = My_Custom_Generator(X_val, y_val, batch_size)
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(28,28,3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same' ))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit_generator(generator=my_training_batch_generator,

                   steps_per_epoch = int(1638 // batch_size),

                   epochs = 10,

                   verbose = 1,

                   validation_data = my_validation_batch_generator,

                   validation_steps = int(183 // batch_size))
#batch_size = 32

#steps = nb_validation_samples / batch_size

#predictions = model.predict_generator(val_generator, steps)



#predicted_classes = convert_to_class(predictions)