# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import cv2
from random import shuffle
from tqdm import tqdm
import os
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = "dogsvscats-{}-{}.model".format(LR, '6conv-basic')

def label_image(img):
    world_label = img.split('.')[0]
    if world_label == "dog": return [1, 0]
    elif world_label == 'cat': return [0, 1]
def create_training_data():
    training_data = []
    for img in os.listdir(TRAIN_DIR):
        label = label_image(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    #np.save("training_data.npy", training_data)
    return training_data
        
        
        
def process_test_data():
    testing_data = []
    for img in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), np.array(img_num)])
    np.save("testing_data.npy", testing_data)
    return testing_data    
training_data = create_training_data()
#training_data = np.load('training_data.npy')
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

train = training_data[: -500]
test = training_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array([i[1] for i in train])
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = np.array([i[1] for i in test])
print(len(X), len(Y), len(test_y), len(test_y))
print(np.shape(X), len(Y), len(test_y), len(test_y))
model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)
#import matplotlib.pyplot as plt
#test_data = process_test_data()
#test_data = np.load("testing_data.npy")
#fig = plt.figure()
#i = 100
#model.load(MODEL_NAME)
#for num, data in enumerate(test_data[i:i+12]):
    #img_num = data[1]
    #img_data = data[0]
    #y = fig.add_subplot(3, 4, num+1)
    #orig = img_data
    #data = img_data.reshape(IMG_SIZE, IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    #if np.argmax(model_out) == 1: str_label = "Cat"
    #else: str_label = "Dog"
    #y.imshow(orig, cmap='gray')    
    #plt.title(str_label)
    #y.axes.get_xaxis().set_visible(False)
    #y.axes.get_yaxis().set_visible(False)
#plt.show()
test_data = process_test_data()
with open("submission-file.csv",'w') as f:
    f.write('id,label\n')
with open("submission-file.csv",'a') as f:
    for num, data in enumerate(test_data):
        if num%1000 == 0: print('write {} line'.format(num))
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num, model_out[0]))
print('done')