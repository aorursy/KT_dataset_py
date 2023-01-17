# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
def getRawFeatures(picture):

    red = []

    green = []

    blue = []

    for row in range(picture.shape[0]):

        for col in range(picture.shape[1]):

            red.append(picture[row][col][0])

            green.append(picture[row][col][1])

            blue.append(picture[row][col][2])

    feature = red

    feature.extend(green)

    feature.extend(blue)

    return feature
import io

import bson # this is installed with the pymongo package

import matplotlib

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

import multiprocessing as mp      # will come in handy due to the size of the data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Simple data processing

count_images = 0

image_names_array = []

category_id_array = []

data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))

pictures = []

count = 0

prod_to_category = dict()



for c, d in enumerate(data):

    #for each product_id

    product_id = d['_id']

    category_id = d['category_id'] # This won't be in Test data

    prod_to_category[product_id] = category_id

    

    for e, pic in enumerate(d['imgs']):

        #for each image

        picture = imread(io.BytesIO(pic['picture']))

        pictures.append(picture)

        count = count + 1

        # do something with the picture, etc

#         image_name = "prod_id-" + str(product_id) + "-" + "image-" + str(e)

#         print("PRODUCT ID:", product_id, "NUMBER", e)

#         plt.imshow(picture)

#         fig1 = plt.gcf()

#         plt.show()

#         plt.draw()

        count_images = count_images + 1

#         image_names_array.append(image_name)

        category_id_array.append(str(category_id))

        #fig1.savefig("img/" + str(image_name), dpi=100)

#     break



prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')

prod_to_category.index.name = '_id'

prod_to_category.rename(columns={0: 'category_id'}, inplace=True)
X_train = np.asarray(pictures)

y_train = np.asarray(category_id_array)

y_train.shape
X_train = X_train.reshape(X_train.shape[0], 3, 180, 180).astype('float32')

X_train = X_train - np.mean(X_train) / X_train.std()
y_train
b,c = np.unique(y_train, return_inverse=True)
from collections import Counter

d = Counter(c)
y_train = c
# y_train
X = X_train

y = y_train

print(X.shape)

print(y.shape)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)



# print(X_train.shape)

# print(y_train.shape)

# print(X_test.shape)

# print(y_test.shape)
from keras.utils import np_utils

from tflearn.data_utils import to_categorical

y_train = np_utils.to_categorical(y_train)

# y_test = np_utils.to_categorical(y_test)
# y_train
from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.estimator import regression

model = input_data(shape=[None,3,180,180])

model = conv_2d(model,32,5,activation='elu')

model = max_pool_2d(model,2)

model = conv_2d(model,64,3,activation='relu')

model = max_pool_2d(model,2)

model = dropout(model,0.3)

model = conv_2d(model,64,3, activation='elu')

model = max_pool_2d(model,2)

model = fully_connected(model,512,activation='sigmoid')

model = dropout(model,0.3)

model = fully_connected(model,36,activation='softmax')

model = regression(model,optimizer='adagrad',loss='categorical_crossentropy',learning_rate=0.05)
import tensorflow as tf

import tflearn

with tf.device('cpu:0'):

   model = tflearn.DNN(model)

   model.fit(X_train , y_train, n_epoch=5, validation_set = (X_train, y_train), batch_size = 10)
#get a few test examples

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=20)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

pred = model.predict(X_test)

pred


# y_test = np_utils.to_categorical(y_test)

# model.evaluate(X_test, y_test)

y_test.shape

# from keras.preprocessing import image

# img = image.load_img('',target_size=(180,180))

# img = image.img_to_array(img)

# img = np.expand_dims(img, axis=0)

img = X_train[0].reshape(1,3,180,180)

# img.shape

preds = model.predict(img)

preds