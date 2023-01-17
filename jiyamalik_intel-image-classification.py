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
# importing the necessary libraries
import tensorflow as tf
import tensorflow.keras.layers as layers 
import tensorflow.keras.activations as activations
import tensorflow.keras.models as models 
import tensorflow.keras.optimizers as optimizers 
import tensorflow.keras.utils as utils
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
import glob
# glob is basically used when we have to retreive the pathnames in some specified pattern 
# In this, glob has been used with wildcard (*) 
# Generally the syntax of glob function is glob.glob(pathname,*,recursive = False)
glob.glob('../input/intel-image-classification/seg_train/seg_train/*',recursive = False) 
# There are total of six classes i.e buildings, glacier , sea, mountain, forest, street 
# copy the path of the train and test folders 
Train_image = "../input/intel-image-classification/seg_train/seg_train"
Test_image = "../input/intel-image-classification/seg_test/seg_test" 
pred_image = "../input/intel-image-classification/seg_pred"
# This will return a list containing names of the images in the sea directory under the seg_train directory
# given by path
sea_images = os.listdir(Train_image + "/sea")
sea_images
# This will give us the total number of images in the sea_images list 
len(sea_images)
# Similarly, this would give me the number of images in the forest_images list 
forest_images = os.listdir(Train_image + "/forest")
len(forest_images)
mountain_images = os.listdir(Train_image + "/mountain")
glacier_images = os.listdir(Train_image + "/glacier")
buildings_images = os.listdir(Train_image + "/buildings")
street_images = os.listdir(Train_image + "/street")
# Let us see what is the total number of training images we have in the dataset
a = len(sea_images) # total number of images in the sea_images list
b = len(forest_images)# total number of images in the forest_images list
c = len(mountain_images)# total number of images in the mountain_images list 
d = len(glacier_images)# total number of images in the glacier_images list
e = len(buildings_images)# total number of images in the buildings_images list 
f = len(street_images)# total number of images in the street_images list
total = a+b+c+d+e+f # Adding the number of images in each class will give us the total training images 
print(total)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Data augmentation 
train_g = ImageDataGenerator(rescale=1./255,  # rescaling the images
                             rotation_range=15, # setting degree range (40) for random rotation 
                             width_shift_range=0.2, # fraction (0.2) of total width
                             height_shift_range=0.2, # fraction (0.2) of total height
                             shear_range=0.2, # shear intensity 
                             zoom_range=0.2, # for zooming inside pictures
                             horizontal_flip=True) #randomly flipping half of the images horizontally

data = ImageDataGenerator(rescale = 1./255) 
pred = ImageDataGenerator(rescale=1./255)   
# generator for training 
# Train_image is the directory where the data is located. It contains subdirectories, 
# each containing images for a class
train_gen = data.flow_from_directory(Train_image, 
                                     target_size = (150,150), # setting the target size as (150,150)
                                     class_mode = "categorical", #setting class_mode as categorical 
                                                                 #because there are more than two
                                                                 #classes to predict  
                                     color_mode = "rgb",  # color_mode is rgb as there are three 
                                                          # coloured channels
                                     shuffle = True) # shuffles the order of the image
# generator for testing 
test_gen = data.flow_from_directory(Test_image,
                                    target_size = (150,150),
                                    class_mode = "categorical",
                                    color_mode = "rgb")
# generator for predictions 
pred_gen = pred.flow_from_directory(
                        pred_image, 
                        shuffle=False,
                        class_mode="categorical",
                        target_size=(150,150)) 
# __getitem__ is used only in indexed attributes like arrays, dictionaries,lists. These provide validation 
# that only correct values are set to the attributes and the only correct caller has access to these attributes
data1 = test_gen.__getitem__(1)[0] 
label = test_gen.__getitem__(1)[1] 
# There are 6 total labels corresponding to each class
l = ["buildings","forest","glacier","mountain","sea","street"] 
# This is for plotting the images along with their labels
plt.figure(figsize = (15,8)) 
i = 17
while i>=0: 
    plt.subplot(4, 6, i+1) # shows multiple images 
    plt.title(l[np.argmax(label[i])]) # labels get shown as title 
    plt.axis('off') # axis would not be present 
    plt.imshow(data1[i]) # displays the images 
    i-=1
import tensorflow as tf 
from tensorflow import keras 
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
model = models.Sequential()  # it allows to create models layer by layer 
# creating two lists dense_layers and conv_layers 
dense_layers = [256, 512]
conv_layers = [128, 256] 
# training the model by using convolutional and pooling layers
# The inputs are 150X150 rgb images 
for dense in dense_layers:  
    for conv in conv_layers:
        model = tf.keras.models.Sequential([tf.keras.layers.Conv2D( conv, (3,3), activation = 'relu',input_shape = (150,150,3))])
        model.add(MaxPooling2D(2,2))
# adding more layers to the model 
model.add(Conv2D( conv, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D( 64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))           
model.summary()   # prints the summary of the model
model.add(Conv2D( 32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))       
model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(dense, activation = 'relu'))
model.add(Dense(6, activation = 'softmax'))
model.summary()
# compiling the model using the optimizer (Adam)
model.compile(optimizer= keras.optimizers.Adam(lr=0.0001),loss='categorical_crossentropy',
              metrics=['accuracy'])  
# fitting the model to train it 
history = model.fit(train_gen, validation_data = test_gen,epochs=25,verbose = 1) 
model.evaluate(test_gen)  # evaluating the accuracy of the model
# making predictions for the model
final = model.predict(pred_gen) 
final  
pred_indices = np.argmax(final,axis=1)  # returns the indices of the maximum values along the axis=1
labels = (train_gen.class_indices)
labels = dict((x,y) for y,x in labels.items())  # creating a dictionary 
predictions = [labels[y] for y in pred_indices] # making predictions 
predictions # displaying the final predictions 