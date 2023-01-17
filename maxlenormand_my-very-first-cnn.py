import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

from PIL import Image



import time



import os

print(os.listdir("../input"))
train_labels = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

test_labels = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')



#print('train : ','\n', train_sample.head(5))

#print('test : ','\n', test_labels.head(5))
test_labels.shape
#This image is labelled as having a cancer cell.

image = plt.imread('../input/histopathologic-cancer-detection/train/c18f2d887b7ae4f6742ee445113fa1aef383ed77.tif')

plt.imshow(image)

plt.show()
image.shape
#let's start with a small sample first:



#size train sample:

x = 30000



#size of val sample:

l = 5000



train_sample = train_labels[:x]

val_sample = train_labels[x:x+l]

test_sample = test_labels
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.utils import layer_utils, to_categorical

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

import tensorflow as tf

from sklearn.metrics import roc_auc_score



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline
img_heigth, img_width = 96, 96
img=load_img('../input/histopathologic-cancer-detection/train/c18f2d887b7ae4f6742ee445113fa1aef383ed77.tif')
train_sample.iloc[0][1]
nb_train_examples=train_sample.shape[0]

nb_val_examples=val_sample.shape[0]



train_img_array = np.ndarray(shape=[nb_train_examples, 96, 96, 3])

train_img_label = np.ndarray(shape=[nb_train_examples, 1])



val_img_array = np.ndarray(shape=[nb_val_examples, 96, 96, 3])

val_img_label = np.ndarray(shape=[nb_val_examples, 1])



test_img_array = np.ndarray(shape=[test_sample.shape[0], 96, 96, 3])

test_img_label = np.ndarray(shape=[test_sample.shape[0], 1])
t1=time.time()

for p in range(nb_train_examples):

    #We turn the .tif into an array

    img_name=train_sample.iloc[p][0]

    img=load_img('../input/histopathologic-cancer-detection/train/'+img_name+'.tif')

    img=img_to_array(img)

    img=img/255

    #print(img_name)

    #print(img.shape)

    train_img_array[p]=img #putting the image inside the 4 dim array

    

    #We put the label into a new ndarray:

    train_img_label[p]=train_sample.iloc[p][1]

t2=time.time()

print('time to turn .tif into array for train_set : ',t2-t1)

print('train_img_array shape is : ', train_img_array.shape)

print('train_img_label shape is : ', train_img_label.shape)
t1=time.time()

for p in range(nb_val_examples):

    #We turn the .tif into an array

    img_name=val_sample.iloc[p][0]

    img=load_img('../input/histopathologic-cancer-detection/train/'+img_name+'.tif')

    img=img_to_array(img)

    img=img/255

    #print(img_name)

    #print(img.shape)

    val_img_array[p]=img #putting the image inside the 4 dim array

    

    #We put the label into a new ndarray:

    val_img_label[p]=val_sample.iloc[p][1]

t2=time.time()

print('time to turn .tif into array for val_sample : ',t2-t1)

print('val_img_array shape is : ', val_img_array.shape)

print('val_img_label shape is : ', val_img_label.shape)
def auroc(y_true, y_pred):

    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)



def FirstModel(num_classes):

    

    model = Sequential()

    model.add(ResNet50(include_top = False, pooling='avg'))

    model.add(Dense(num_classes, activation = 'sigmoid'))

    

    model.layers[0].trainable = False

    

    return model
#my_model=FirstModel(train_img_array[0].shape)

my_model = FirstModel(num_classes = 2)
my_model.compile(optimizer = Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy', auroc])
train_img_label = to_categorical(train_img_label, num_classes=2)

val_img_label = to_categorical(val_img_label, num_classes=2)
stats = my_model.fit(x = train_img_array, y = train_img_label, epochs = 5)
evaluation = my_model.evaluate(x= val_img_array, y=val_img_label)

print()

print ("Loss = " + str(evaluation[0]))

print ("Test Accuracy = " + str(evaluation[1]))
#This turned out to be a bad idea. But I'm keeping it, never know when I might need it



dummy_img = np.ndarray(shape=(1, 96, 96, 3))



t1=time.time()

for p in range(test_sample.shape[0]):

    #We turn the .tif into an array

    img_name=test_sample.iloc[p][0]

    img=load_img('../input/histopathologic-cancer-detection/test/'+img_name+'.tif')

    img=img_to_array(img)

    img=img/255

    

    #print(img_name)

    #print(img.shape)

    

    pred = my_model.predict(img.reshape(1,96,96,3))

    #We put the label into a new ndarray:

    test_sample.at[p ,'label'] = pred.argmax()

    

    

t2=time.time()

print('time to turn .tif into array for test_sample : ',t2-t1)

print('number of images labelled with cancer : ',test_sample[test_sample['label']==1].shape[0],

      ' out of ', test_sample.shape[0], ' examples')
test_sample.to_csv('test_predictions.csv', index=False)