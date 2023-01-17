from numpy.random import seed
seed(888)
import tensorflow
tensorflow.random.set_seed(404)
import numpy as np
import tensorflow as tf

from tensorflow import keras

from IPython.display import display

from keras.preprocessing.image import array_to_img


import matplotlib.pyplot as plt

%matplotlib inline

from keras.datasets import cifar10
(x_train_all , y_train_all),(x_test,y_test) = cifar10.load_data()
# type(cifar10)
type(x_train_all)
LABEL_NAMES = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_PIXCELS = IMAGE_WIDTH*IMAGE_HEIGHT
COLOR_CHANNELS = 3
TOTAL_INPUT = IMAGE_PIXCELS * COLOR_CHANNELS

VALIDATION_SIZE = 10000
x_train_all.shape
x_train_all[0].shape
x_train_all[7]
type(x_train_all[0][0][0][0])
pic = array_to_img(x_train_all[13])
display(pic)
y_train_all.shape
y_train_all[13][0]
LABEL_NAMES[y_train_all[13][0]]
plt.imshow(x_train_all[4])
plt.xlabel(LABEL_NAMES[y_train_all[4][0]])
plt.show()
plt.figure(figsize=(15,5))

for i in range(10):
  plt.subplot(1,10,i+1)
  plt.yticks([])
  plt.xticks([])
  plt.xlabel(LABEL_NAMES[y_train_all[i][0]],fontsize=14)
  plt.imshow(x_train_all[i])
x_train_all[0].shape
x_train_all.shape
nr_images, x ,y , c = x_train_all.shape
print(f'images = {nr_images} \t | width = {x} \t | height = {y} \t | channels = {c}')
type(x_train_all[0][0][0][0])
x_train_all[0][0][0][0]
x_train_all , x_test = x_train_all/255.0 , x_test/255.0
x_train_all[0][0][0][0]
x_train_all.shape[0]
32*32*3
x_train_all = x_train_all.reshape(x_train_all.shape[0],TOTAL_INPUT)
x_train_all.shape
len(x_test)
x_test = x_test.reshape(len(x_test),TOTAL_INPUT)
print(f"Shape of x_test is {x_test.shape}")
x_train_all.shape
x_val = x_train_all[:VALIDATION_SIZE]
y_val = x_train_all[:VALIDATION_SIZE]
x_val.shape
y_val.shape


