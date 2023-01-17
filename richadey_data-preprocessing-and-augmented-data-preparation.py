# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/data-2/test.csv", dst = "../working/test.csv"'')

copyfile(src = "../input/data-2/train.csv", dst = "../working/train.csv")

copyfile(src = "../input/process-and-utils/Data_Preprocessing.py", dst = "../working/Data_Preprocessing.py")

copyfile(src = "../input/process-and-utils/Utils_funX.py", dst = "../working/Utils_funX.py")

copyfile(src = "../input/brightness-and-sharpness-augmented-data2/Brightness_And_Sharpness_Augmented_data2.ipynb", dst = "../working/Brightness_And_Sharpness_Augmented_data2.ipynb")
!pip install plantcv
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from Utils_funX import *

from Data_Preprocessing import *

from keras.layers import *
%env JOBLIB_TEMP_FOLDER=/tmp
train_data = pd.read_csv("train.csv")

test_data = pd.read_csv("test.csv")
#Since we have no Null values in the dataset--> we only remov the Duplicates

train_data = remove_duplicates(train_data)

test_data = remove_duplicates(test_data)

print("\nThe shape of the Traning samples after Data Preprocessing is {} \n".format(train_data.shape))

print("\nThe shape of the Testing samples after after Data Preprocessing is {} \n".format(test_data.shape))
x_train,y_train,x_test,y_test = Data_Preparation(train_data,test_data)
import gc

del train_data

del test_data

gc.collect()
print("\nThe train and test data Shapes are :",x_train.shape,y_train.shape,x_test.shape,y_test.shape)
x_train,x_test=Data_Normalization(x_train,x_test)
width = 48

height = 48
x_train = x_train.reshape(x_train.shape[0],width,height,)

x_test = x_test.reshape(x_test.shape[0],width,height,)
x_train.shape
from keras.utils import to_categorical



y_train = to_categorical(y_train,7)

print(y_train.shape)

y_train
y_test = to_categorical(y_test,7)

print(y_test.shape)

y_test
x_train[0]
x_train.shape,x_test.shape
!pip3 install import_ipynb
import cv2

from PIL import Image

from keras.preprocessing.image import array_to_img

from keras.preprocessing.image import img_to_array
x_t = []



for i in range(len(x_train)):

    x = x_train[i]

    x = array_to_img(np.reshape(x,(48,48,1)))

    x = x.resize((96,96))

    x = np.reshape(img_to_array(x),(96,96))

    

    x_t.append(x)

x_train = np.array(x_t)



x_train.shape
del x

del x_t

gc.collect()
x_t = []



for i in range(len(x_test)):

    x = x_test[i]

    x = array_to_img(np.reshape(x,(48,48,1)))

    x = x.resize((96,96))

    x = np.reshape(img_to_array(x),(96,96))

    

    x_t.append(x)

x_test = np.array(x_t)



x_test.shape
del x_t

del x

gc.collect()
import import_ipynb

from Brightness_And_Sharpness_Augmented_data2 import *
# adding two type of data 1 -> vertically flipped and horizontally fliped

# therefore After augmentation



x_train1,y_train1=Brightness_And_Sharpness_Augmented_data(x_train,y_train,len(x_train))

x_train2,y_train2=Data_Augmentation(x_train,y_train)



x_train = np.concatenate((x_train1,x_train2)) 

y_train = np.concatenate((y_train1,y_train2))
del x_train1

del y_train1

del x_train2

del y_train2

gc.collect()
x_train.shape, y_train.shape
x_train.shape, y_train.shape
#Storing the final Augmented X and Y values in the  a csv file

from numpy import savez_compressed

savez_compressed('x_train.npz',x_train)

savez_compressed('x_test.npz',x_test)

savez_compressed('y_train.npz',y_train)

savez_compressed('y_test.npz',y_test)