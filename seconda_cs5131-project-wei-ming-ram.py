# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import PIL.Image
import cv2
import re
import matplotlib.pyplot as plt
import random
import math

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPool2D
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#this function orders the images using natural sort
#normal sort order: img01, img135, img296, img37
#natural sorted order: img01, img37, img296, img135
def natural_key(astr):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', astr)]

#helper function to get an ndarray from image files
def load_image_array(directory):
    dfa = []
    filelist = sorted(os.listdir(directory), key=natural_key)
    for filename in filelist:
        path = directory + '/' + filename
        image = PIL.Image.open(path) # open colour image
        image = np.array(image)
        image = image[:, :, :3] #remove transparency layer (alpha)
        dimension=(200,200) #specify number of pixels
        rimage = cv2.resize(image, dimension, interpolation=cv2.INTER_AREA) #resize images to specified size
        dfa.append(rimage)
    dfnew = np.array(dfa) #convert list to ndarray
    
    return dfnew

# get loss for training set and validation set
# plot for each epoch
def plot_loss(num, history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model ' + str(num) + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

#haversine function to calculate distance
#shortest distance above earth's surface
def haversine(n1, w1, n2, w2):
    earthrad = 6371000
    phi_1 = math.radians(n1)
    phi_2 = math.radians(n2)
    phi_change = math.radians(n2 - n1)
    lambda_change = math.radians(w1 - w2) #(reversed because west is negative)
    
    a = math.sin(phi_change/2) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(lambda_change/2) ** 2
    c = 2 * math.atan2(a ** 0.5, (1 - a) ** 0.5)
    d = earthrad * c
    return d

#distance between top left and bottom right coordinates of area covered (maximum distance)
print(haversine(40.57770678, 73.74077701, 40.90365244, 74.03151652))
#start actual code
trainDirectory = '../input/train-images/trainimg'
testDirectory = '../input/test-images/imagestest'
trainX = load_image_array(trainDirectory)
testX = load_image_array(testDirectory)
trainY = pd.read_csv('../input/train-coordinates/traincoord.txt', sep=" ", header=None)
trainY.columns = ['North', 'West']
testY = pd.read_csv('../input/test-coordinates/testcoord.txt', sep=" ", header=None)
testY.columns = ['North', 'West']

maxNorth = trainY['North'].max()
minNorth = trainY['North'].min()
maxWest = trainY['West'].max()
minWest = trainY['West'].min()
print(maxNorth)
print(minNorth)
print(maxWest)
print(minWest)
randnum = random.randrange(400)
if randnum < 10:
    sample_image_path = trainDirectory + '/img0' + str(randnum) + '.PNG'
else:
    sample_image_path = trainDirectory + '/img' + str(randnum) + '.PNG'
sample_image = PIL.Image.open(sample_image_path)
plt.figure()
plt.imshow(sample_image) 
plt.show()  # display it
print(trainY.iloc[randnum])

#note that we have used the original image to show here, so the pixel size is different from (200,200)
model1 = Sequential()
filters = [16, 32, 64]

for f in filters:
    model1.add(Conv2D(f, kernel_size=7, activation='relu', input_shape=(200,200,3), padding='same'))
    model1.add(Conv2D(f, kernel_size=3, activation='relu', input_shape=(200,200,3), padding='same'))
    model1.add(BatchNormalization())
    model1.add(MaxPool2D((2,2)))

model1.add(Flatten())
model1.add(Dense(3, activation='relu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.25))
model1.add(Dense(2, activation='relu'))
model2 = Sequential()
filters = [16, 32, 64]

for f in filters:
    model2.add(Conv2D(f, kernel_size=3, activation='relu', input_shape=(200,200,3), padding='same'))
    model2.add(BatchNormalization())
    model2.add(MaxPool2D((2,2)))

model2.add(Flatten())
model2.add(Dense(3, activation='relu'))
model2.add(BatchNormalization())
model2.add(Dropout(0.25))
model2.add(Dense(2, activation='relu'))
model3 = Sequential()
filters = [16, 32, 64]

for f in filters:
    model3.add(Conv2D(f, kernel_size=3, activation='relu', input_shape=(200,200,3), padding='same'))
    model3.add(BatchNormalization())
    model3.add(Conv2D(f, kernel_size=3, strides=(2,2), activation='relu', input_shape=(200,200,3), padding='same'))

model3.add(Flatten())
model3.add(Dense(3, activation='relu'))
model3.add(BatchNormalization())
model3.add(Dropout(0.25))
model3.add(Dense(2, activation='relu'))
#scaling coordinates from zero to one for better performance
maxNorth = trainY['North'].max()
minNorth = trainY['North'].min()
diffNorth = maxNorth - minNorth
maxWest = trainY['West'].max()
minWest = trainY['West'].min()
diffWest = maxWest - minWest

trainY['North'] = (trainY['North'] - minNorth) / diffNorth
trainY['West'] = (trainY['West'] - minWest) / diffWest
testY['North'] = (testY['North'] - minNorth) / diffNorth
testY['West'] = (testY['West'] - minWest) / diffWest

es1 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
es2 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
es3 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model1.compile(loss='mean_squared_error', optimizer='adam')
model2.compile(loss='mean_squared_error', optimizer='adam')
model3.compile(loss='mean_squared_error', optimizer='adam')
history1 = model1.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=10, callbacks=[es1])
history2 = model2.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=10, callbacks=[es2])
history3 = model3.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=10, callbacks=[es3])
# get loss for training set and validation set
# plot for each epoch    
plot_loss(1, history1)
plot_loss(2, history2)
plot_loss(3, history3)
pred1 = model1.predict(testX)
pred2 = model2.predict(testX)
pred3 = model3.predict(testX)
pred1[:,0] = pred1[:,0] * diffNorth + minNorth
pred1[:,1] = pred1[:,1] * diffWest + minWest
pred2[:,0] = pred2[:,0] * diffNorth + minNorth
pred2[:,1] = pred2[:,1] * diffWest + minWest
pred3[:,0] = pred3[:,0] * diffNorth + minNorth
pred3[:,1] = pred3[:,1] * diffWest + minWest
testY['North'] = testY['North'] * diffNorth + minNorth
testY['West'] = testY['West'] * diffWest + minWest
#use haversine formula to calculate distance
dist1 = np.empty(100)
dist2 = np.empty(100)
dist3 = np.empty(100)
for count in range(100):
    dist1[count] = haversine(pred1[count,0], pred1[count,1], testY['North'].iloc[count], testY['West'].iloc[count])
    dist2[count] = haversine(pred2[count,0], pred2[count,1], testY['North'].iloc[count], testY['West'].iloc[count])
    dist3[count] = haversine(pred3[count,0], pred3[count,1], testY['North'].iloc[count], testY['West'].iloc[count])

print(dist1[:5])
print(dist2[:5])
print(dist3[:5])

avg1 = dist1.mean()
avg2 = dist2.mean()
avg3 = dist3.mean()

std1 = dist1.std()
std2 = dist2.std()
std3 = dist3.std()

print(str(avg1) + ' ' + str(std1))
print(str(avg2) + ' ' + str(std2))
print(str(avg3) + ' ' + str(std3))