# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/dogs-vs-cats/train/train/"))

# Any results you write to the current directory are saved as output.
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = '../input/dogs-vs-cats/train/train/'
TEST_DIR = '../input/testing/tiny_test/tiny_test/'
IMG_SIZE = 50
LR = 0.001
MODEL_NAME = 'dogs-vs-cats-cnn'
def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), create_label(img)])
    shuffle(testing_data)
    np.save('testing_data.npy', testing_data)
    return testing_data


if (os.path.exists('train_data.npy')): # If you have already created the dataset:
    train_data =np.load('train_data.npy')
    #train_data = create_train_data()
else: # If dataset is not created:
    train_data = create_train_data()

if (os.path.exists('test_data.npy')):
    test_data =np.load('test_data.npy')
else:
    test_data = create_test_data()


train = train_data
test = test_data
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

tf.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)

#Student code
conv6 = conv_2d(pool5, 64, 5, activation='relu')
pool6 = max_pool_2d(conv6, 5)
#Student code

fully_layer = fully_connected(pool5, 784, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 2, activation='softmax')
cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)


#if (os.path.exists('model.tfl.meta')):
#    model.load('./model.tfl')

#else:
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
      batch_size = 32,
      validation_set=({'input': X_test}, {'targets': y_test}),
      snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save('model.tflearn')

acc = 0
model.load("model.tflearn")
for i in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, i)
        img = cv2.imread(path,0)
        test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict([test_img])[0]
        if(i.startswith('dog') and np.max(prediction) == prediction[1]):
            acc+=1
        elif(i.startswith('cat') and np.max(prediction) == prediction[0]):
            acc+=1
        print(np.max(prediction))
        print(f"cat: {prediction[0]}, dog: {prediction[1]}")
print("Accuracy = ",acc/20*100,"%")
