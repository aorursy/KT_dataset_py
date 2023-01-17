import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

from sklearn.utils import shuffle

from keras.models import Sequential

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D

from keras.layers import MaxPooling2D, Dropout

from sklearn.model_selection import KFold

from keras.models import Model

from PIL import Image

import cv2

import IPython

print(os.listdir('../input'))
print(os.listdir('../input/objects'))
photo_array=[]

target=[]

size=100

from os import listdir

from matplotlib import image

loaded_images = list()

for filename in listdir('../input/objects/glass'):

    image = Image.open('../input/objects/glass/' + filename)

    if (image.mode != 'RGB'):

        image=image.convert('RGB')

    img_resized = np.array(image.resize((size,size)))

    photo_array.append(img_resized) 

    target.append(0)

    

for filename in listdir('../input/objects/water_bottle'):

    if (image.mode != 'RGB'):

        image=image.convert('RGB')

    img_resized = np.array(image.resize((size,size)))

    photo_array.append(img_resized) 

    target.append(1)

    

for filename in listdir('../input/objects/chair'):

    image = Image.open('../input/objects/chair/' + filename)

    if (image.mode != 'RGB'):

        image=image.convert('RGB')

    img_resized = np.array(image.resize((size,size)))

    photo_array.append(img_resized) 

    target.append(2)
X=np.array(photo_array)

y=np.array(target)

X=X/255
X,y=shuffle(X,y,random_state=44)
from sklearn.preprocessing import OneHotEncoder

hot = OneHotEncoder()

y=y.reshape(len(y), 1)

y = hot.fit_transform(y).toarray()
kfold = KFold(n_splits=10, shuffle=False, random_state=22)

cvscores = []

for train, test in kfold.split(X, y):

    size=X.shape[1]

    model = Sequential()

    model.add(ZeroPadding2D(2, input_shape=(size, size, 3)))

    model.add(Conv2D(32, (7, 7),strides=(1, 1),padding="valid", kernel_initializer='random_uniform',

                bias_initializer='zeros',activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),strides=(1, 1),padding="valid", activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3,3),strides=(1, 1),padding="valid", activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(256, (1,1),strides=(1, 1),padding="valid", activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(3, activation="softmax")) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X[train], y[train],batch_size = 128, epochs=30,verbose=0 )

    scores = model.evaluate(X[test], y[test], verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))  