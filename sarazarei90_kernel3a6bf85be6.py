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
import tensorflow as tf
import os 

from tqdm import tqdm

import cv2

import numpy as np





def load_data(data_path):

    labels = []

    images = []

    paths = []

    size = 128, 128

    

    for folder in tqdm(os.listdir(data_path), position=0):

        for file in tqdm(os.listdir(os.path.join(data_path, folder)), position=0):

            if file.endswith("jpg"):

                img = cv2.imread(os.path.join(data_path, folder, file))

                im = cv2.resize(img, size)



                paths.append(os.path.join(data_path, folder, file))

                labels.append(folder)

                images.append(im)

            else:

                continue

        

    return np.array(images), labels, paths

    

train_images, train_labels, train_paths = load_data("../input/training/training/")

test_images, test_labels, test_paths = load_data("../input/validation/validation/")



print('Train:', train_images.shape, len(train_labels))

print('Test:', test_images.shape, len(test_labels))
import matplotlib as mpl

import matplotlib.pyplot as plt



%matplotlib inline
ulabels = np.unique(train_labels).tolist()



train_labels_d = [ulabels.index(label) for label in train_labels]

test_labels_d = [ulabels.index(label) for label in test_labels]
print(train_labels[0], train_labels_d[0])
x_train = x_train / 255.

x_test = x_test / 255.
plt.imshow(x_train[0])
plt.imshow(x_test[0])
np.random.choice(5, 3)
x_train, y_train = train_images, train_labels_d

x_test, y_test = test_images, test_labels_d

x_train = x_train / 255.

x_test = x_test / 255.
from sklearn.utils import shuffle





x_train, y_train = shuffle(x_train, y_train)

x_test, y_test = shuffle(x_test, y_test)
def build_model(dnn_units):

    l1=tf.keras.regularizers.l1(0.001)

    model, r, evaluate = None, None, None

    

    model =tf.keras.Sequential([

        #c2d - #1

        

        tf.keras.layers.Conv2D(16, (3,3),input_shape = (128,128,3)),

        tf.keras.layers.Dropout(rate=0.4),

        tf.keras.layers.MaxPool2D((2,2)),

#         tf.keras.layers.Dropout(rate = 0.4)(x),

        #c2d - #2

        tf.keras.layers.Conv2D(32, (3,3),activation = "tanh",activity_regularizer=l1),

        tf.keras.layers.Dropout(rate=0.4),

        tf.keras.layers.MaxPool2D((2,2)),

#         tf.keras.layers.Dropout(rate = 0.1)(x),

#         #c2d - #3

        tf.keras.layers.Conv2D(64,(3,3),activation = "tanh",activity_regularizer=l1),

        tf.keras.layers.Dropout(rate=0.1),

        tf.keras.layers.MaxPool2D((2,2)),



        #flatten

        tf.keras.layers.Flatten(),

        #fc

        tf.keras.layers.Dense(dnn_units, activation="tanh",activity_regularizer=l1),

        tf.keras.layers.Dropout(rate=0.1),

        

        #softmax

        tf.keras.layers.Dense(10, activation="softmax"),

    ])

    

    opt = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(optimizer='adam' , loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])



    r = model.fit(x_train , np.array(y_train), validation_split=0.2, epochs = 20 , batch_size = 128 , verbose = 1),



    evaluate = model.evaluate(x_test, np.array(y_test) , verbose = 0),

    return model, r, evaluate

  
model, r, evaluate = build_model(64)

evaluate