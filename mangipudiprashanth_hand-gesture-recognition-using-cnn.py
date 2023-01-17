# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from sklearn.utils import shuffle
# Please change the directory path according to your path.
#Load Images from Swing
loadedImages = []
for i in range(0, 1000):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/SwingImages/swing_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From Palm
for i in range(0, 1000):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/PalmImages/palm_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images From Fist
for i in range(0, 1000):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/FistImages/fist_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
#Load Images From One
for i in range(0, 1000):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/One/one_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From Two
for i in range(0, 1000):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/Two/two_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From Three
for i in range(0, 1000):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/Three/three_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
    

# Create OutputVector

outputVectors = []
for i in range(0, 1000):
    outputVectors.append([1, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 1, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 1, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 1, 0, 0])    

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 1, 0])    
    
for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 1])    

testImages = []

#Load Images for swing
for i in range(0, 100):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/SwingTest/swing_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images for Palm
for i in range(0, 100):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/PalmTest/palm_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images for Fist
for i in range(0, 100):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/FistTest/fist_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
#Load Images for One
for i in range(0, 100):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/OneTest/one_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
#Load Images for Two
for i in range(0, 100):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/TwoTest/two_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
#Load Images for Three
for i in range(0, 100):
    image = cv2.imread('/kaggle/input/hand-gesture-recognition/Dataset/ThreeTest/three_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))    

#One-hot encoding of test images    
testLabels = []

for i in range(0, 100):
    testLabels.append([1, 0, 0,0,0,0])
    
for i in range(0, 100):
    testLabels.append([0, 1, 0,0,0,0])

for i in range(0, 100):
    testLabels.append([0, 0, 1,0,0,0])
for i in range(0, 100):
    testLabels.append([0, 0, 0,1,0,0]) 
for i in range(0, 100):
    testLabels.append([0, 0, 0,0,1,0])
for i in range(0, 100):
    testLabels.append([0, 0, 0,0,0,1])    
# Define the CNN Model
tf.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,6,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Shuffle Training Data
loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)

# Train model
model.fit(loadedImages, outputVectors, n_epoch=50,
           validation_set = (testImages, testLabels),
           snapshot_step=100, show_metric=True, run_id='Beast')

model.save("TrainedModel/GestureRecogModel.tfl")