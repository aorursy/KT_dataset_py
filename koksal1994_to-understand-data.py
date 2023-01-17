import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os # accessing directory structure
import sys
import random
%matplotlib inline
import tensorflow as tf
import keras_preprocessing
from tensorflow.keras.preprocessing import image
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
tf.__version__
# Any results you write to the current directory are saved as output.
TRAINING_DIR = '/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/' 
VALIDATION_DIR = '/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/'

# this is the augmentation configuration we will use for training
train_gen = ImageDataGenerator(rescale = 1./255)
valid_gen = ImageDataGenerator(rescale = 1./255)
 
#rgb --> the images will be converted to have 3 channels.

train_data = train_gen.flow_from_directory(
TRAINING_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb",
batch_size=1
)

valid_data = valid_gen.flow_from_directory(
VALIDATION_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb",
batch_size=1
)
#
for cl_indis, cl_name in enumerate(train_data.class_indices):
     print(cl_indis, cl_name)
#train_data is a 'DirectoryIterator' yielding tuples of (X, y). 
#We dont use batch
print(len(train_data)) #18345 picture
print("**********************************************************")
print(len(train_data[0])) # this is a tupple (X,y)-->
print("**********************************************************")
print(len(train_data[0][0])) # return lists of [X] --> The lenth is (1x227,227,3), 1 is batch size, (227,227) is input size of image and 3 is channel number(RGB)
print(train_data[0][0].shape)
print("**********************************************************")
print(len(train_data[0][0][0])) #X at i. batch --> (227,227,3)
print(train_data[0][0][0].shape)
print("**********************************************************")
print(train_data[0][0][0][0].shape) # return a row
plt.imshow(train_data[0][0][0][5]) #plot 5.row
print("**********************************************************")
# we plot between 0-50 row
plt.imshow(train_data[0][0][0][0:50])
plt.show()

print(train_data[0][1].shape) # y, The lenth is (128,10)
print("**********************************************************")
print(train_data[0][1][0]) #get 0.picture label

def f_class_by_array(cl_arr):
    cl = 0
    for i in range(len(cl_arr)):
        if cl_arr[i] == 1:
            cl_name = f_class_name_by_cl(cl)
            return cl_name
        else:
            cl += 1
            
def f_class_name_by_cl(cl):
    for cl_name, cl_ind in train_data.class_indices.items():
        if cl_ind == cl:
            return cl_name
plt.figure(figsize=(20,10))
for i in range(5): #first 5 images
    plt.subplot(5/5+1, 5, i+1)
    cl_name = f_class_by_array(train_data[i][1][0])
    plt.title("Class:{}".format(cl_name))
    plt.imshow(train_data[i][0][0])




