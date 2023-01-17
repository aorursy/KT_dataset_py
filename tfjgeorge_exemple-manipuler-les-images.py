# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load images with numpy
images_train = np.load('../input/train_images.npy', encoding='latin1')
images_train.shape
#Load labels
train_labels = np.genfromtxt('../input/train_labels.csv', names=True, delimiter=',', dtype=[('Id', 'i8'), ('Category', 'S5')])
train_labels.shape
#Reshaping image to 100x100
image_train1 = (images_train[3][1]).reshape(100,100)
plt.imshow(image_train1)
#Printing label
print(train_labels[3])
#Load images with numpy
images_test = np.load('../input/test_images.npy', encoding='latin1')
images_test.shape
#Reshaping image to 100x100
image_test1 = (images_test[0][1]).reshape(100,100)
plt.imshow(image_test1)