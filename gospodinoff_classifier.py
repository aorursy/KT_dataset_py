# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
print(os.listdir("../input"))
import pickle
# Any results you write to the current directory are saved as output.

from skimage import data, img_as_float
from skimage import exposure
from skimage.color import rgb2gray
from keras.utils import to_categorical

X = []
Y = []
indices = [1,3,4,0,32,8,7,2]
cats = [0,1,2,3,4,5,6,7]
train_data = pickle.load(open('../input/train.p','rb'))
val_data = pickle.load(open('../input/train.p','rb'))
count = 0;
# train_data['features'] = 0.299 * train_data['features'][:, :, :, 0] + 0.587 * train_data['features'][:, :, :, 1] + 0.114 * train_data['features'][:, :, :, 2];

for index,label in enumerate(train_data['labels']):
    img = train_data['features'][index]
#     img = rgb2gray(img)
#     img = exposure.equalize_hist(img)
#     cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    X.append(img)
    Y.append(label)
    count+=1
        
        
plt.imshow(X[10])
plt.show()
X = np.array(X)/255.
# X = X.reshape(count,32,32,1);
Y = to_categorical(Y, num_classes=43)
print(Y)
X_val = []
Y_val = []
indices = [1,3,4,0,32,8,7,2]
cats = [0,1,2,3,4,5,6,7]
val_data = pickle.load(open('../input/valid.p','rb'))
count = 0;

# val_data['features'] = 0.299 * val_data['features'][:, :, :, 0] + 0.587 * val_data['features'][:, :, :, 1] + 0.114 * val_data['features'][:, :, :, 2];

for index,label in enumerate(val_data['labels']):
    img = val_data['features'][index]
#     img = rgb2gray(img)
#     img = exposure.equalize_hist(img)
#     cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    X_val.append(img)
    Y_val.append(label)
    count+=1
        
        
plt.imshow(X_val[1000])
plt.show()
X_val = np.array(X_val)/255.
# X_val = X_val.reshape(count,32,32,1);
Y_val = to_categorical(Y_val, num_classes=43)
print(Y_val)
l = []
for index,label in enumerate(train_data['labels']):
    if(label not in l):
        l.append(label)
        img = train_data['features'][index]
        plt.imshow(img)
        print(label)
        plt.show()
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense,Activation,ZeroPadding2D,Convolution2D
from keras.optimizers import adam, sgd, rmsprop

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(32,32,3)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

optimizer = adam(lr=0.0001)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])
model.summary()
model.fit(X,Y,validation_split=0.2,epochs=10, validation_data=(X_val, Y_val), shuffle=True)
# model.fit_generator(datagen.flow(X, y=Y, batch_size=32), epochs=10)
model.save('model.h5')
from keras.models import load_model
test_data = pickle.load(open('../input/test.p','rb'))

mod = load_model('../model.h5')
mod.evaluate(X_val, Y_val)
print(len(test_data['labels']))
im_index = 3659
testim = test_data['features'][im_index];
plt.imshow(testim)
pred = mod.predict(np.array([testim]))
print('PREDICTION - '+ str(np.argmax(pred[0]) )+ '\nGROUND TRUTH - '+str(test_data['labels'][im_index]))