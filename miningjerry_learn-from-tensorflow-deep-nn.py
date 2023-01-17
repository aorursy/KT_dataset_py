# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



import tensorflow as tf



#settings

LEARNING_RATE = 1e-4

TRAINING_ITERATIONS =2500

DROPOUT=0.5

BATCH_SIZE = 50

VALIDATION_SIZE = 2000

IMAGE_TO_DISPLAY = 10
data = pd.read_csv('../input/train.csv')

print('data({0[0]},{0[1]})'.format(data.shape))

print(data.head())
images = data.iloc[:,1:].values

images = images.astype(np.float)

print(images[20:])

# convert from [:255] to [0.1,1.0]

images = np.multiply(images,1.0/255.0)

print(images[20:])

print('images({0[0]},{0[1]})'.format(images.shape))
image_size = images.shape[1]

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print('Change imgage size {0} => {1} X {2}'.format(image_size, image_width, image_height))
# Display image

def display(img):

    one_image=img.reshape(image_width, image_height)

    

    plt.axis('off')

    plt.imshow(one_image, cmap=cm.binary)



display(images[IMAGE_TO_DISPLAY])
labels_flat = data[[0]].values.ravel()

print ('labels_flat({0})'.format(len(labels_flat)))

print('lables_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels_flat[IMAGE_TO_DISPLAY]))
labels_count = np.unique(labels_flat).shape[0]

print('labels_count => {0}'.format(labels_count))
# convert class labels from scalars to one-hot vectors

# 0 => [1 0 0 0 0 0 0 0 0 0]

# 1 => [0 1 0 0 0 0 0 0 0 0]

# ...

# 9 => [0 0 0 0 0 0 0 0 0 1]

def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] =1

    return labels_one_hot



labels = dense_to_one_hot(labels_flat, labels_count)

labels = labels.astype(np.uint8)



print('labels({0[0]},{0[1]})'.format(labels.shape))

print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))
#np.flat,np.ravel(), np.flatten()

#np.flat 和np.ravel（）是一样的，修改源数据

a = np.zeros((3,4))

b = np.zeros((3,4))

c = np.zeros((3,4))

e = np.zeros((3,4))

print(a)

for i in range(12):

    a.flat[i] = i

    b.ravel()[i] =i

    c.flatten()[i] = i

    

    

    

print(a)

print(a.ravel())

print(b)

print(c)