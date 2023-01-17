# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing librariles 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,MaxPooling2D
from keras.utils import np_utils
%matplotlib inline

import cv2 #resizing images
import os 

img_width = 68
img_height = 68
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'
# complete dataset 
#this contains both dog and cat paths 
#creating paths
train_images=[]
for i in os.listdir(TRAIN_DIR):
    train_images.append(TRAIN_DIR+i)
train_images
# trainig data set for dogs individually 
dog_train_images=[]
for i in os.listdir(TRAIN_DIR):
    if 'dog' in i:
        dog_train_images.append(TRAIN_DIR+i)
    
dog_train_images
# all data paths will be stored in this dog_train_images 
cats_train_images=[]
for i in os.listdir(TRAIN_DIR):
    if 'cat' in i:
        cats_train_images.append(TRAIN_DIR+i)
cats_train_images
#all data paths of cats will be stored in cats_train_images
# creating paths and storing in list
test_images=[]
for i in os.listdir(TEST_DIR):
    test_images.append(TEST_DIR+i)
test_images
# Insted of taking whole set of images we will take sample data pf cats and dogs, buils the model
#taking 500 images of cats amd 500 images of dogs and we will build the model
cat_images=cats_train_images[:500]
dog_images=dog_train_images[:500]

Number_of_cat_images=np.array(cat_images).shape
Number_of_dog_images=np.array(dog_images).shape
print(Number_of_cat_images)
print(Number_of_dog_images)
train_image= cat_images+dog_images
np.array(train_image).shape
test_image=test_images[:50]
# Now we have paths ... we should convert them to matrices by using cv2 lib and convert image to grey scale
# small example of enumerate 
#this is not related to this project 
#I just wanna show how enuerate works
lis=['a','b','c']

for i,j in enumerate(lis):
    print(i,j)

def prepare_data(list_of_images):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    x = [] # images as arrays
    y = [] # labels
    
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (img_width,img_height), interpolation=cv2.INTER_CUBIC))
    
    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
        #else:
            #print('neither cat nor dog name present in images')
            
    return x, y
X, Y = prepare_data(train_images)
np.array(X).shape
X1=np.array(X)
X1.shape
X1=X1.reshape(25000,4624)
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.flow((X1),Y,batch_size=32)
classifier.fit_generator(train_generator,epochs=30)
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)

X, Y = prepare_data(train_images_dogs_cats)
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, 64, 64,3), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        return data

train = prep_data(train_images)
test = prep_data(test_images)
train.shape
labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)

train
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(train, labels, batch_size=16, epochs=10,
              validation_split=0.25, verbose=0, shuffle=True)
mod = model.predict(test)
predictions



