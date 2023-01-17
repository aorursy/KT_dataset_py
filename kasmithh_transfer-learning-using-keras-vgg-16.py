import numpy as np
import pandas as pd
import os
import numpy
import glob
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense
print(os.listdir("../input/flowers-recognition/flowers/flowers"))
weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
#https://stackoverflow.com/questions/49537604/how-to-read-multiple-images-from-multiple-folders-in-python
folders = glob.glob('../input/flowers-recognition/flowers/flowers/*')
imagenames_list = []
for folder in folders:
    for f in glob.glob(folder+'/*.jpg'):
        imagenames_list.append(f)
def label_img(image):
    word_label = image.split('/')[4]
    if word_label == 'daisy':
        return [1,0,0,0,0]
    elif word_label == 'dandelion':
        return [0,1,0,0,0]
    elif word_label == 'rose':
        return [0,0,1,0,0]
    elif word_label == 'tulip':
        return [0,0,0,1,0]
    else:
        return [0,0,0,0,1]
train = []        

for image in imagenames_list:
    label = label_img(image)
    train.append([np.array(cv2.resize(cv2.imread(image),(224,224))), np.array(label)])
    np.random.shuffle(train)
X = np.array([i[0] for i in train])
X = X/255
Y = np.array([i[1] for i in train])
model = Sequential()
model.add(VGG16(include_top = False, weights = weights, input_shape = (224,224,3)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
Fit = model.fit(X, Y, epochs = 1, validation_split = 0.30)