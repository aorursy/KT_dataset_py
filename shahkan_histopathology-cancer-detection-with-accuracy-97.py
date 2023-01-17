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
import keras.backend as k

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

import tensorflow as tf

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from glob import glob

from skimage.io import imread
df = pd.DataFrame({'path': glob(os.path.join('../input/train', '*.tif'))})

df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0])

labels = pd.read_csv('../input/train_labels.csv')

df = df.merge(labels, on="id")

df.head()
df0 = df[df.label == 0].sample(500)

df1 = df[df.label == 1].sample(500)

df = pd.concat([df0, df1])

df = df[["path", "id", "label"]]

df.shape
df['image'] = df['path'].map(imread)

df.head()
image = (df['image'][450], df['label'][450])

img = plt.imshow(image[0])
input_images = np.stack(list(df.image), axis=0)

input_images.shape

Y = LabelBinarizer().fit_transform(df.label)

X = input_images
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
def model(input_shape):

    # Defining the input placeholder

    X_input = Input(input_shape)

    

    # Padding the borders

    X = ZeroPadding2D((3, 3))(X_input)

    # Applying the first layer

    X = Conv2D(32, (7, 7), strides= (1, 1), name='conv0')(X)

    X = BatchNormalization(axis=3, name='bn0')(X)

    X = Activation('relu')(X)

     # MaxPool

    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    

    # Applying the second layer

    X = Conv2D(64, (7, 7), strides= (1, 1), name='conv1')(X)

    X = BatchNormalization(axis=3, name='bn1')(X)

    X = Activation('relu')(X)

    # MaxPool

    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    

    #Applying third layer

    X = Conv2D(128, (7, 7), strides= (1, 1), name='conv2')(X)

    X = BatchNormalization(axis=3, name='bn2')(X)

    X = Activation('relu')(X)

     # MaxPool

    X = MaxPooling2D((2, 2), name='max_pool3')(X)  

    # Flatten and FullyConnected Layer

    X = Flatten()(X)

    X = Dense(1, activation='sigmoid', name='fc')(X)

    

    model = Model(inputs=X_input, outputs=X, name='Model')

    

    return model



model_final = model(train_X.shape[1:])
model_final.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model_final.fit(train_X, train_Y, epochs=10, batch_size=50)
evals = model_final.evaluate(test_X, test_Y, batch_size=32, verbose=1)

print('Test accuracy: '+str(evals[1]*100)+'%')