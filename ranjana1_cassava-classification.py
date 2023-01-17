# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

from matplotlib import pyplot as plt

import cv2

image=cv2.imread('../input/train/train/cmd/train-cmd-1992.jpg')

plt.imshow(image)

plt.show()
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

first_gray = cv2.GaussianBlur(gray, (5, 5), 0)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal

kernel = np.ones((3,3),np.uint8)

opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)



# sure background area

sure_bg = cv2.dilate(opening,kernel,iterations=3)



# Finding sure foreground area

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)



# Finding unknown region

sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_fg)

plt.imshow(sure_bg)

#plt.imshow(first_gray)

plt.show()

# Marker labelling

ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1

markers = markers+1

# Now, mark the region of unknown with zero

markers[unknown==255] = 0

plt.imshow(markers)

plt.show()

markers = cv2.watershed(image,markers)

image[markers == -1] = [255,0,0]

plt.imshow(image)

plt.show()

#edges = cv2.Sobel(image,cv2.CV_64F,1,0, ksize=1)

edges=cv2.Canny(image,200,300)

plt.subplot(121),plt.imshow(first_gray,cmap = 'gray')

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(edges,cmap = 'gray')

plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
np.mean(edges)

np.mean(image)
import cv2 

import os 

import numpy as np 

from random import shuffle 

from tqdm import tqdm 
TRAIN_DIR = '../input/train/train/cgm'

TEST_DIR = '../input/test/test/0'

#IMG_SIZE = 96

LR = 1e-3

MODEL_NAME = 'cassava-{}-{}.model'.format(LR, '6conv-basic') 

class_c=''



import numpy.ma as ma

def create_train_data(): 

 training_data = [] 

 for img in tqdm(os.listdir(TRAIN_DIR)): 

             # labeling the images 

            #label = label_img(img) 

   # grayc=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #edgedet=cv2.Canny(np.array(grayc),200,300,10)

    #m=np.ma.mean(img)

    path = os.path.join(TRAIN_DIR, img) 

              # loading the image from the path and then converting  into gray

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

              # resizing the image 

    img = cv2.resize(img, (100,100)) 

              # forming the training data list with numpy array of the images 

            #training_data.append([np.array(img), np.array(label)]) 

    label=os.path.basename(TRAIN_DIR)

    training_data.append([np.array(np.mean(img)), np.array(label)])

            

               # shuffling of the training data to preserve the random state of our data 

shuffle(training_data) 

         # saving our trained data for further uses if required 

np.save('train_data.npy', training_data) 

print(*training_data, sep=",")

return training_data 

print(os.path.basename(TRAIN_DIR))
import numpy.ma as ma

def process_test_data(): 

    testing_data = [] 

    for img in tqdm(os.listdir(TEST_DIR)): 

         #edge_test=cv2.Canny(np.asarray(img),200,300)

         path = os.path.join(TEST_DIR, img) 

         img_num = img.split('.')[0] 

         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

         img = cv2.resize(img, (666, 500)) 

         testing_data.append([np.array(np.ma.mean(img)), np.array(img_num)]) 

         

    shuffle(testing_data) 

    np.save('test_data.npy', testing_data) 

    return testing_data 
#executing model

train_data = create_train_data() 

test_data = process_test_data() 
import tflearn 

from tflearn.layers.conv import conv_2d, max_pool_2d 

from tflearn.layers.core import input_data, dropout, fully_connected 

from tflearn.layers.estimator import regression 

  

import tensorflow as tf 

tf.reset_default_graph() 

convnet = input_data(shape =[None, 666, 500, 1], name ='input') 

  

convnet = conv_2d(convnet, 32, 5, activation ='relu') 

convnet = max_pool_2d(convnet, 5) 

  

convnet = conv_2d(convnet, 64, 5, activation ='relu') 

convnet = max_pool_2d(convnet, 5) 
convnet = conv_2d(convnet, 128, 5, activation ='relu') 

convnet = max_pool_2d(convnet, 5) 

  

convnet = conv_2d(convnet, 64, 5, activation ='relu') 

convnet = max_pool_2d(convnet, 5) 

  

convnet = conv_2d(convnet, 32, 5, activation ='relu') 

convnet = max_pool_2d(convnet, 5) 
convnet = fully_connected(convnet, 1024, activation ='relu') 

convnet = dropout(convnet, 0.8) 
convnet = fully_connected(convnet, 2, activation ='softmax') 

convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 

      loss ='categorical_crossentropy', name ='targets') 
model = tflearn.DNN(convnet, tensorboard_dir ='log')