import numpy as np
import os
import re
import cv2
import h5py
from random import shuffle
from tqdm import tqdm
import keras
from keras.callbacks import TensorBoard
import tensorflow as tf
print(cv2.__version__)
#print(os.listdir("C:/Users/User/Desktop/all/train"))
train_dir = "../input/train/train/"
test_dir = "../input/test1/test1/"
IMG_SIZE = 32
LR = 1e-3

MODEL_NAME = 'dc-{}-{}.model'.format(LR, '2conv-basic')

img_width = 32
img_height = 32
input_shape = (img_width, img_height, 1)
def labled(name):
    if "cat" == name.split(".")[0]:return[0, 1]
    elif "dog" == name.split(".")[0]:return[1, 0]
def trains():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = labled(img)
        path = os.path.join(train_dir,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
def tests():
    testing_data = []
    for img in os.listdir(test_dir):
        label = labled(img)

        path = os.path.join(test_dir,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
train_data = trains()
test_data =tests()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D


model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(32,32,1)))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
from matplotlib import pyplot as plt
train = train_data[:-500]
test = test_data[:-500]
X = np.array([i[0] for i in train])
X = X.reshape(X.shape[0],IMG_SIZE,IMG_SIZE,1)
Y = np.array([i[1] for i in train])
X = X.astype('float32')
X /= 255
print(Y.shape)
print(Y)

test_x = np.array([i[0] for i in test])
test_x = test_x.reshape(-1,IMG_SIZE,IMG_SIZE,1)
# test_x= test_x.reshape(test_x.shape[0],IMG_SIZE,IMG_SIZE,1)
print(test_x.shape)
test_x = test_x.astype('float32')
test_x /= 255
test_y = np.array([i[1] for i in test])
test_y=test_y.reshape(test_y.shape[0],1)
print(test_y.shape,test_x.shape)


%%time
import tensorflow as tf
from keras_tqdm import TQDMNotebookCallback
with tf.device("/device:GPU:0"):
    model.fit(X,Y, batch_size=32, epochs=50, verbose=2, callbacks=[TQDMNotebookCallback()], validation_split=0.2)
    

%%time
import copy
with tf.device('/gpu:0'):
    model.save(filepath = "my_model3.h5")
results = model.predict_classes(test_x)
print(results)

print(results[1])
plt.imshow(test_x[1].reshape(32,32))



