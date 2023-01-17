# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

bee_data = pd.read_csv("../input/honey-bee-pollen/pollendataset/PollenDataset/pollen_data.csv")



import glob, os 

from skimage import io, transform

import numpy as np

import matplotlib.pyplot as plt

path="../input/honey-bee-pollen/pollendataset/PollenDataset/images/"

imlist= glob.glob(os.path.join(path, '*.jpg'))

def dataset(file_list,size=(300,180),flattened=False):

	data = []

	for i, file in enumerate(file_list):

		image = io.imread(file)

		image = transform.resize(image, size, mode='constant')

		if flattened:

			image = image.flatten()



		data.append(image)



	labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in file_list]



	return np.array(data), np.array(labels)
X,y=dataset(imlist)

# X has the following structure: X[imageid, y,x,channel]

print('The length of X: ',len(X))  # data

print('The shape of X: ',X.shape)  # target

print('The shape of Y', y.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(

	X, y, test_size=0.20)



partial_x_train, validation_x_train, partial_y_train, validation_y_train = train_test_split(

	x_train, y_train, test_size=0.15)

from keras import layers

from keras import models

from keras import optimizers



model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(300,180,3)))  #input shape must be the match the input image tensor shape

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64,(3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128,(3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128,(3,3), activation='relu'))

model.add(layers.Conv2D(128,(3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(256,(3,3), activation='relu'))

model.add(layers.Conv2D(256,(3,3), activation='relu'))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))



model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
history = model.fit(

    partial_x_train, 

    partial_y_train,

    validation_data=(validation_x_train, validation_y_train),

    epochs=50, 

    batch_size=15, 

    verbose =2) #hides some information while training
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

test_loss, test_acc = model.evaluate(x_test, y_test, steps=10)

print('The final test accuracy: ',test_acc)