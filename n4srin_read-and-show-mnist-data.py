import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
training_images_filepath = '/kaggle/input/mnist-dataset/train-images.idx3-ubyte'
training_labels_filepath = '/kaggle/input/mnist-dataset/train-labels.idx1-ubyte'
test_images_filepath = '/kaggle/input/mnist-dataset/t10k-images.idx3-ubyte'
test_labels_filepath = '/kaggle/input/mnist-dataset/t10k-labels.idx1-ubyte'
X_train, y_train = loadlocal_mnist( training_images_filepath, training_labels_filepath)
X_test, y_test = loadlocal_mnist(test_images_filepath, test_labels_filepath)
print('Dimension of training images:', np.shape(X_train))
print('Dimension of trainig labels:',np.shape(y_train))
print('Dimension of testing images:', np.shape(X_test))
print('Dimension of testing labels:',np.shape(y_test))
# reshape the arrays:
X_train = X_train.reshape(60000, 28, 28)
X_test = X_test.reshape(10000, 28, 28)
print('Dimension of training images:', np.shape(X_train))
print('Dimension of testing images:', np.shape(X_test))
def show(image, title):
    index = 1 
    plt.figure(figsize=(10,5))

    for x in zip(image, title):        
        image = x[0]        
        title = x[1]
        plt.subplot(2, 5, index)        
        plt.imshow(image, cmap=plt.cm.gray)  
        plt.title(x[1], fontsize = 9)
        index += 1
image = []
title = []
for i in range(0, 5):
    r = random.randint(1, len(X_train))
    image.append(X_train[r])
    title.append('training image:' + str(y_train[r]))       

for i in range(0, 5):
    r = random.randint(1, len(X_test))
    image.append(X_test[r])
    title.append('testing image:' + str(y_test[r]))
    
show(image, title)