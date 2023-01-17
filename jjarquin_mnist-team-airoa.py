# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.cm as cm



from pandas import Series, DataFrame

from sklearn.decomposition import PCA

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import tensorflow as tf

LEARNING_RATE = 1e-4

TRAINING_ITERATIONS = 2500



DROPOUT = 0.5

BATCH_SIZE = 50



VALIDATION_SIZE = 2000



IMAGE_TO_DISPLAY = 10
data = pd.read_csv('../input/train.csv')

data.head()


images = data.iloc[:,1:].values

images = images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]

print ('image_size => {0}'.format(image_size))



# in this case all images are square

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)



print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))
# display image

def display(img):

    

    # (784) => (28,28)

    one_image = img.reshape(image_width,image_height)

    

    plt.axis('off')

    plt.imshow(one_image, cmap=cm.binary)



# output image     

display(images[IMAGE_TO_DISPLAY])

# get the unique labels

labels_flat = data[[0]].values.ravel()

labels_count = np.unique(labels_flat).shape[0]



def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot  

labels = dense_to_one_hot(labels_flat, labels_count)

labels = labels.astype(np.uint8)


# split data into training & validation

validation_images = images[:VALIDATION_SIZE]

validation_labels = labels[:VALIDATION_SIZE]



train_images = images[VALIDATION_SIZE:]

train_labels = labels[VALIDATION_SIZE:]
# weight initialization

def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)


# convolution

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# input & output of NN



# images

x = tf.placeholder('float', shape=[None, image_size])

# labels

y_ = tf.placeholder('float', shape=[None, labels_count])