# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np 

import itertools

import keras

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 

from keras.models import Sequential, Model, load_model

from keras.preprocessing import image

from keras.utils.np_utils import to_categorical 

import matplotlib.pyplot as plt 

import matplotlib.image as mpimg

%matplotlib inline

import math 

import datetime

import time

import cv2

import os

import tqdm

import shutil

from skimage.io import imread

from skimage.transform import resize



from keras.models import load_model

from sklearn.datasets import load_files   

from keras.utils import np_utils

from glob import glob

from keras import applications

from keras.preprocessing.image import ImageDataGenerator 

from keras import optimizers

from keras.models import Sequential,Model,load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D, BatchNormalization

from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint



import tensorflow as tf

from tensorflow.keras.applications import InceptionResNetV2

from tensorflow.keras.models import Sequential
dest_dir = "/kaggle/input/shopee-coranteam-product-detection-batch/all_images/all_images"

kaggle_dir = "/kaggle/input/shopee-coranteam-product-detection-batch/"

#Load training samples and labels

filenames_shuffled = np.load(kaggle_dir+'filenames_shuffled.npy')

y_labels_one_hot_shuffled = np.load(kaggle_dir+'y_labels_one_hot_shuffled.npy')
# Used this line as our filename array is not a numpy array.

filenames_shuffled_numpy = np.array(filenames_shuffled)



X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(

    filenames_shuffled_numpy, y_labels_one_hot_shuffled, test_size=0.2, random_state=1)



print(X_train_filenames.shape)

print(y_train.shape)          



print(X_val_filenames.shape)  

print(y_val.shape)          



img_height,img_width = 224, 224
class My_Custom_Generator(keras.utils.Sequence) :

  

    def __init__(self, image_filenames, labels, batch_size) :

        self.image_filenames = image_filenames

        self.labels = labels

        self.batch_size = batch_size





    def __len__(self) :

        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)



    def __getitem__(self, idx) :

        def Preprocessing(img):

            

            ####################################################################

            out_img = np.zeros((img_height, img_width, 3))

            

            #out_img = resize(img, (img_height, img_width, 3)) #<<= This is insanely slow 

            out_img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)

            ####################################################################

            

            return np.array(out_img)

        

        batch_x_img = []

        batch_y_img = []

        

        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]



        batch_x_img =  np.array([

                Preprocessing(cv2.imread(dest_dir + "/" + str(file_name), cv2.COLOR_BGR2RGB))

                   for file_name in batch_x])/255.0

        batch_y_img = np.array(batch_y)



        return batch_x_img, batch_y_img
#Creating instances 

batch_size = 256 #1024 #512 #256 #128 #32 



my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train, batch_size)

my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val, batch_size)
NUM_CLASSES = 42



IMAGE_RESIZE = 224



#Initialization of model

model = Sequential()



model.add(InceptionResNetV2(include_top = False, pooling = 'avg', weights = 'imagenet'))



model.add(Flatten())



model.add(Dense(2048, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Dense(NUM_CLASSES, activation='softmax'))



model.layers[0].trainable = False



model.summary()



opt = keras.optimizers.Adam(learning_rate=0.003)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(generator=my_training_batch_generator,

                   steps_per_epoch = int(3800 // batch_size),

                   epochs = 20,

                   verbose = 1,

                   validation_data = my_validation_batch_generator,

                   validation_steps = int(950 // batch_size))
# plot the loss and accuracy

import matplotlib.pyplot as plt

%matplotlib inline



acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.title('Training and validation accuracy')

plt.plot(epochs, acc, 'red', label='Training acc')

plt.plot(epochs, val_acc, 'blue', label='Validation acc')

plt.legend()



plt.figure()

plt.title('Training and validation loss')

plt.plot(epochs, loss, 'red', label='Training loss')

plt.plot(epochs, val_loss, 'blue', label='Validation loss')



plt.legend()



plt.show()
model.save('saved_model/my_model')
new_model.predict
