# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



print("Hello")
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
np.fromstring(b'\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@', dtype='<f4')
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

        image_names_array.append(image_name)

        category_id_array.append(str(category_id))

        #fig1.savefig("img/" + str(image_name), dpi=100)

    print("done")

#     break



prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')

prod_to_category.index.name = '_id'

prod_to_category.rename(columns={0: 'category_id'}, inplace=True)
X_train = np.asarray(pictures)

y_train = np.asarray(category_id_array)

y_train.shape
X_train.shape
X_train = X_train.reshape(X_train.shape[0], 3, 180, 180).astype('float32')

X_train = X_train - np.mean(X_train) / X_train.std()
# listFeatureVector = []



# for picture in pictures:

#     featureVector = getRawFeatures(picture)

#     listFeatureVector.append(featureVector)
# print(len(listFeatureVector[0]))



# X = listFeatureVector

# y = category_id_array
# len(list(set(y)))
y_train
b,c = np.unique(y_train, return_inverse=True)
from collections import Counter

d = Counter(c)
y_train = c
y_train
from keras.utils import np_utils

from tflearn.data_utils import to_categorical

y_train = np_utils.to_categorical(y_train)
y_train