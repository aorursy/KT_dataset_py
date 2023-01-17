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
import tensorflow as tf

import os

import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from tensorflow.keras.preprocessing import image

import PIL
%cd ../input/

os.listdir()
!ls

!pwd
os.listdir('waste-classification-data/dataset/DATASET/')
print('TRAIN R IS ', len(os.listdir('waste-classification-data/dataset/DATASET/TRAIN/R')))

print('TRAIN O IS ', len(os.listdir('waste-classification-data/dataset/DATASET/TRAIN/O')))

print('TEST R IS ', len(os.listdir('waste-classification-data/dataset/DATASET/TEST/R')))

print('TEST O IS ', len(os.listdir('waste-classification-data/dataset/DATASET/TEST/O')))
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten 

from keras.layers import Dense
classifier= Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(224,224,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(output_dim = 128,activation='relu'))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

base=classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('waste-classification-data/dataset/DATASET/TRAIN',target_size=(224,224),batch_size=32,class_mode='binary')

test_set=test_datagen.flow_from_directory('waste-classification-data/dataset/DATASET/TEST',target_size=(224,224),batch_size=32,class_mode='binary')
from IPython.display import display

from PIL import Image

hist=classifier.fit_generator(training_set,epochs=7,validation_data=test_set)
import numpy as np

from keras.preprocessing import image

test_image=image.load_img('waste-classification-data/DATASET/TEST/O/O_12577.jpg',target_size=(224,224))

test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict(test_image)

training_set.class_indices

if result[0][0] >= 0.8:

    prediction='O'

else:

    prediction='R'

print(prediction)

    
%matplotlib inline

accuracy = hist.history['accuracy']

loss = hist.history['loss']

val_accuracy = hist.history['val_accuracy']

val_loss = hist.history['val_loss']



epochs = range(len(accuracy))



plt.plot(epochs, accuracy, 'r', "Training Accuracy")

plt.plot(epochs, val_accuracy, 'b', "Testing Accuracy")

plt.title('Training vs Testing Accuracy')

plt.figure()



plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Testing Loss")

plt.title('Training vs Testing Loss')

plt.show()