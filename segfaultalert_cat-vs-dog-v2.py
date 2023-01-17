# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re 
import cv2 
import glob
import matplotlib.pyplot as plt 
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm 
from random import shuffle
import os
train = os.listdir("../input/train")
test  = os.listdir("../input/test")
train_dir = "../input/train"
test_dir = "../input/test"
Height = 50 
Width = 50 
cwd = os.getcwd() 

# Any results you write to the current directory are saved as output.
def get_label(img):
    label = img.split('.')[-3] 
    if label == 'cat':
        return [1,0]
    elif label == 'dog':
        return [0,1] 
def process_train_data():
    train_data = [] 
    path = os.path.join(cwd, train_dir,'*g')
    imgs = glob.glob(path)
    for img in tqdm(imgs):
        labels = get_label(img.split('/')[6])  
        img = cv2.imread(img,0) 
        img = cv2.resize(img,(Height,Width)) 
        train_data.append([np.array(img), np.array(labels)])
    
    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data    
def process_test_data():
    test_data = [] 
    path = os.path.join(cwd, test_dir,'*g')
    imgs = glob.glob(path)
    for img in tqdm(imgs):
        img_idx = img.split('/')[6].split('.')[0]
        img = cv2.imread(img,0)
        img = cv2.resize(img,(Height,Width))
        test_data.append([np.array(img), img_idx])
    shuffle(test_data)
    np.save('test_data.npy', test_data)
    return test_data 
train_data = process_train_data()
#Model 
convnet = input_data(shape=[None, Height, Width, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)
train = train_data[:-500]
validation = train_data[-500:]
X = np.array([i[0] for i in train]).reshape(-1,Height,Width,1)
Y = [i[1] for i in train] 
X_val = np.array([i[0] for i in validation]).reshape(-1,Height,Width,1)
Y_val = [i[1] for i in validation]
#training model 
model.fit({'input': X}, {'targets': Y}, n_epoch=15, validation_set=({'input': X_val}, {'targets': Y_val}), 
    snapshot_step=500, show_metric=True)
test_data = process_test_data() 

#competeing!! 
test_data = process_test_data() 
test_data = np.load('test_data.npy')

with open('submission_file.csv','w') as f:
    f.write('id,label\n')
            
with open('submission_file.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(Height,Width,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))

