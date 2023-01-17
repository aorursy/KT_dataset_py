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
from PIL import Image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import tensorflow as tf

from tensorflow.python.framework import ops

import math

import glob

# from skimage.transform import resize   # for resizing images
#reading image values in pixels width * hight * channels

ext = ['jpg', 'jpeg']    # Add image formats here

data = []

labels = []



files = []

imdir = '../input/train/train/cbb/'

[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

data.extend([cv2.imread(file) for file in files])

labels.extend(["cbb" for file in files])



files = []

imdir = '../input/train/train/cbsd/'

[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

data = np.concatenate([data, [cv2.imread(file) for file in files]])

labels.extend(["cbsd" for file in files])



files = []

imdir = '../input/train/train/cgm/'

[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

data = np.concatenate([data, [cv2.imread(file) for file in files]])

labels.extend(["cgm" for file in files])





files = []

imdir = '../input/train/train/cmd/'

[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

data = np.concatenate([data, [cv2.imread(file) for file in files]])

labels.extend(["cmd" for file in files])



files = []

imdir = '../input/train/train/healthy/'

[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

data = np.concatenate([data, [cv2.imread(file) for file in files]])

labels.extend(["healthy" for file in files])

size = (300, 300)

data = np.array([cv2.resize(d, size, interpolation = cv2.INTER_AREA) for d in data])
# data = np.concatenate([data, [np.fliplr(data[i]) for i in range(len(data))]])

# labels.extend([labels[i] for i in range(len(labels))])
labels = np.array(labels)
lr = [np.fliplr(data[i]) for i in range(len(data))]

labels_lr = [labels[i] for i in range(len(labels))]
data = np.concatenate([data, lr])

labels = np.concatenate([labels, labels_lr])
# data = np.array(data)

labels = np.array(labels)
labels = pd.get_dummies(labels)
X_train = data

Y_train = labels
# #using only training data

# X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=.05, random_state=42, stratify=labels)
#using real test data

testData = []

files = []

imdir = '../input/test/test/0/'

[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

testData.extend([cv2.imread(file) for file in files])



size = (300, 300)

testData = [cv2.resize(d, size, interpolation = cv2.INTER_AREA) for d in testData]



X_train = data

Y_train = labels

X_test = np.array(testData)

testLabels = files
import numpy as np

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



%matplotlib inline
input_shape = ((300, 300, 3))

X_input = Input(input_shape)

    

# Zero-Padding: pads the border of X_input with zeroes

X = ZeroPadding2D((3, 3))(X_input)



# CONV -> BN -> RELU Block applied to X

X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)

X = BatchNormalization(axis = 3, name = 'bn0')(X)

X = Activation('relu')(X)



# MAXPOOL

X = AveragePooling2D((2, 2), name='avg_pool0')(X)



# CONV -> BN -> RELU Block applied to X

X = Conv2D(32, (5, 5), strides = (1, 1), name = 'conv1')(X)

X = BatchNormalization(axis = 3, name = 'bn1')(X)

X = Activation('relu')(X)



# MAXPOOL

X = AveragePooling2D((2, 2), name='avg_pool1')(X)



# CONV -> BN -> RELU Block applied to X

X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv2')(X)

X = BatchNormalization(axis = 3, name = 'bn2')(X)

X = Activation('relu')(X)



# MAXPOOL

X = AveragePooling2D((2, 2), name='avg_pool2')(X)



# FLATTEN X (means convert it to a vector) + FULLYCONNECTED

X = Flatten()(X)

X = Dense(1024, activation="relu")(X)

X = Dropout(0.5)(X)

X = Dense(5, activation='softmax', name='fc')(X)



# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

model = Model(inputs = X_input, outputs = X, name='satellite')
model.summary()
model.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
model.fit(x=X_train, y=Y_train, epochs=50, batch_size=32)
# preds = model.evaluate(x=X_test, y=Y_test)

# print()

# print ("Loss = " + str(preds[0]))

# print ("Test Accuracy = " + str(preds[1]))
pred = model.predict(x=X_test)
pred = [np.argmax(pred[i]) for i in range(len(pred))]
for i in range(len(pred)):

    if pred[i] == 0:

        pred[i] = 'cbb'

    if pred[i] == 1:

        pred[i] = 'cbsd'

    if pred[i] == 2:

        pred[i] = 'cgm'

    if pred[i] == 3:

        pred[i] = 'cmd'

    if pred[i] == 4:

        pred[i] = 'healthy'
testLabels = [l[21:] for l in testLabels]
submission = pd.DataFrame(list(zip(pred, testLabels)), columns=["Category", "Id"])
submission.to_csv('submission.csv', index=False)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(submission)