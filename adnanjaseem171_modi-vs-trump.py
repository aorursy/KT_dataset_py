# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
TRAIN_DIR1 = "../input/modi123/narendra modi/Narendra Modi"
TRAIN_DIR2 = "../input/donald123/donald trump/Donald Trump"
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'modi-vs-trump'

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR1)):
        path = os.path.join(TRAIN_DIR1, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data),[0,1]])
    for img in tqdm(os.listdir(TRAIN_DIR2)):
        path = os.path.join(TRAIN_DIR2, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data),[1,0]])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
    
train_data = create_train_data()
train = train_data[:-100]
test = train_data[-100:]
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]
tf.reset_default_graph()
convnet = input_data(shape= [None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet,4)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet,4)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet,4)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet,4)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet,4)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer= 'adam', learning_rate=LR, loss='categorical_crossentropy')
model = tflearn.DNN(convnet, tensorboard_dir='log',tensorboard_verbose=0)
model.fit(X_train,y_train,n_epoch=17,
         validation_set=(X_test,y_test),
          snapshot_step=500, show_metric=True,run_id=MODEL_NAME)
test_data = []
i = 0
for img in tqdm(os.listdir(TRAIN_DIR1)):
        path = os.path.join(TRAIN_DIR1, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        test_data.append([np.array(img_data),[0,1]])
        i=i+1
        if i > 24:
            break
for img in tqdm(os.listdir(TRAIN_DIR2)):
        path = os.path.join(TRAIN_DIR2, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        test_data.append([np.array(img_data),[1,0]])
        i=i+1
        if i > 49:
            break
shuffle(test_data)
len(test_data)                  
count=0 
print("Model Output ---------------------------------->   Expected Output")
for i in range(50):
    x=model.predict(np.array(test_data[i][0]).reshape(-1,IMG_SIZE, IMG_SIZE, 1))
    if x[0][0]>x[0][1]:
        x=[1,0]
    else:
        x=[0,1]
    print(x, "                                           ", test_data[i][1])
    if x==test_data[i][1]:
        count=count+1
print("accuracy=",(count/50)*100,"%")
        
