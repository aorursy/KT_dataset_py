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
# Importing necessary libraries and mounting drive

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from imageio import imread



import keras

from keras import backend as k



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, Activation



from keras.utils import to_categorical

from keras.datasets import mnist



from skimage.transform import resize

from sklearn.model_selection import train_test_split



import os



from google.colab import drive

drive.mount('/content/dgdrive')

from google.colab import files



df_train = pd.read_csv('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/training.csv')

#df_test = pd.read_csv('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/sample.csv')



# dropping unnecessary columns

df_train = df_train.drop(columns=['Cone Latitude', 'Cone Longitude'])



# 2814 training images

# image size: 1920x1080

# reduce by factor of 4: 480x270

# reduce by factor of 6: 320x180

# reduce by factor of 8: 240x135



img_width, img_height = 480, 270

df_train.head()
train_id = [str(df_train['Id'][i]) for i in range(0,500)]

training_images_temp = [imread('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/TrainingImages//' + j) for j in train_id]

resized_images = [resize(i, (img_width, img_height)) for i in training_images_temp]

training_images = np.array(resized_images)

del train_id

del training_images_temp

del resized_images



train_id = [str(df_train['Id'][i]) for i in range(500,1000)]

training_images_temp = [imread('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/TrainingImages//' + j) for j in train_id]

resized_images = [resize(i, (img_width, img_height)) for i in training_images_temp]

training_images = np.concatenate((training_images, np.array(resized_images)), axis=0)

del train_id

del training_images_temp

del resized_images



train_id = [str(df_train['Id'][i]) for i in range(1000,1500)]

training_images_temp = [imread('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/TrainingImages//' + j) for j in train_id]

resized_images = [resize(i, (img_width, img_height)) for i in training_images_temp]

training_images = np.concatenate((training_images, np.array(resized_images)), axis=0)

del train_id

del training_images_temp

del resized_images



train_id = [str(df_train['Id'][i]) for i in range(1500,2000)]

training_images_temp = [imread('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/TrainingImages//' + j) for j in train_id]

resized_images = [resize(i, (img_width, img_height)) for i in training_images_temp]

training_images = np.concatenate((training_images, np.array(resized_images)), axis=0)

del train_id

del training_images_temp

del resized_images



train_id = [str(df_train['Id'][i]) for i in range(2000,2500)]

training_images_temp = [imread('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/TrainingImages//' + j) for j in train_id]

resized_images = [resize(i, (img_width, img_height)) for i in training_images_temp]

training_images = np.concatenate((training_images, np.array(resized_images)), axis=0)

del train_id

del training_images_temp

del resized_images
# setting training inputs

x_train = training_images[:2500, :, :, :]

y_train = np.array(df_train['Distance'].values)

y_train = y_train[:2500]

# building sequential model

input_shape = (img_width, img_height, 3)



model = Sequential()



model.add(Conv2D(32, kernel_size = (9, 9), 

                 activation = 'relu',

                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, kernel_size = (7, 7), 

                 activation = 'relu',

                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(16, kernel_size = (5, 5), 

                 activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(16, kernel_size = (3, 3), 

                 activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(16, activation='relu'))

#model.add(Dropout(0.1))

model.add(Dense(16, activation='relu'))

#model.add(Dropout(0.25))

model.add(Dense(1, activation = 'linear'))



model.compile(loss='mean_squared_error',

              optimizer='adam',

              metrics=['accuracy'])
model.fit(x_train, 

          y_train, 

          batch_size = 128,

          epochs = 73,

          verbose = 1)
# Testing images resizing 



df_test = pd.read_csv('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/sample.csv')



test_id = [str(df_test['Id'][i]) for i in range(0,399)]

testing_images_temp = [imread('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/TestingImages//' + j) for j in test_id]

resized_images = [resize(i, (img_width, img_height)) for i in testing_images_temp]

testing_images = np.array(resized_images)

del test_id

del testing_images_temp

del resized_images



test_id = [str(df_test['Id'][i]) for i in range(399,703)]

testing_images_temp = [imread('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/TestingImages//' + j) for j in test_id]

resized_images = [resize(i, (img_width, img_height)) for i in testing_images_temp]

testing_images = np.concatenate((testing_images, np.array(resized_images)), axis=0)

del test_id

del testing_images_temp

del resized_images
# Prediciting distnaces on testing images

x_test = testing_images[:703, :, :, :]

y_test = model.predict(x_test)

df_test['Distance'] = y_test

# Saving model and output file

df_test.to_csv('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/solution_AW1.csv', index = False)

np.save('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/training_images.npy', training_images)

model.save('/content/dgdrive/Shared drives/InfoSci-Cones-Distance/autonobot_trial_model_er_8.h5')