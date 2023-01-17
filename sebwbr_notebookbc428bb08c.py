# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import tensorflow as tf

import matplotlib.cm as cm



# settings

LEARNING_RATE = 1e-4

TRAINING_ITERATIONS = 2500        

  

DROPOUT = 0.5

BATCH_SIZE = 50



VALIDATION_SIZE = 2000



IMAGE_TO_DISPLAY = 1000



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# read training data from CSV file 

data = pd.read_csv('../input/train.csv')



print('data({0[0]},{0[1]})'.format(data.shape))

print (data.head())
images = data.iloc[:,1:].values

images = images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)



print('images({0[0]},{0[1]})'.format(images.shape))

image_size = images.shape[1]

print ('image_size => {0}'.format(image_size))



# in this case all images are square

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)



print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))

def display(img):

    

    # (784) => (28,28)

    one_image = img.reshape(image_width,image_height)

    

    plt.axis('off')

    plt.imshow(one_image, cmap=cm.binary)



# output image     

display(images[IMAGE_TO_DISPLAY])
labels_flat = data[[0]].values.ravel()



print('labels_flat({0})'.format(len(labels_flat)))

print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))



labels_count = np.unique(labels_flat).shape[0]



print('labels_count => {0}'.format(labels_count))


