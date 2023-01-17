# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from random import shuffle
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from tqdm import tqdm
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
TRAIN_DIR = "../input/train" #Reading in the training directory.
#TEST_DIR = "../input/test"
IMG_SIZE = 100
def get_label(img):                      #Function to get the label of the image, is it a cat or a dog.
    word_label = img.split('.')[0] 
    if word_label == 'cat' : return [1,0]
    elif word_label == 'dog' : return [0,1]     #Return the respective label.
def create_train_data():                  #Function for creation of the training data
    train_data = []
    for img in (os.listdir(TRAIN_DIR)):        #For each image repeat.
        label = get_label(img)                 #Get the image label.
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))  #Resizing the image
        train_data.append([np.array(img),np.array(label)])
    shuffle(train_data)
    return train_data
train_data = create_train_data()
print(len(train_data))
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2,activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2,activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2,activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2,activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2,activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2,activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024,activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet,2,activation='softmax')
convnet = regression(convnet,optimizer = 'adam', loss='categorical_crossentropy',name='conv_NN')

model = tflearn.DNN(convnet)
train = train_data[:-500]
test = train_data[-500:]
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE,1) 
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE,1) 
test_y = [i[1] for i in test]
model.fit(X,Y, n_epoch= 5,validation_set=(test_x,test_y),snapshot_step=500,show_metric=True,run_id='ConvNN')
import matplotlib.pyplot as plt

fig = plt.figure()

for num, data in enumerate(test[:12]):
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE,1)
    
    model_out = model.predict([data])[0]
    
    if( np.argmax(model_out) == 1):
        str_label = 'Dog'
    else:
        str_label = 'Cat'
    
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
    
    