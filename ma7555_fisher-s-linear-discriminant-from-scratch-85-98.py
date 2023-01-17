# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load Training Images:



im_train = np.loadtxt('../input/train.csv', delimiter=',', dtype=int, skiprows=1)

train_labels = im_train[:, 0]

im_train = im_train[:, 1:]

nclasses = len(np.unique(train_labels))

nfeatures = np.size(im_train, axis=1)

class_indexes = []

for i in range(nclasses):

    class_indexes.append(np.argwhere(train_labels == i))
im_train.shape
# Initializing needed variables

class_means, other_class_means = np.empty((nclasses, nfeatures)), np.empty((nclasses, nfeatures))

other_class = []

SW_one, SW_two, SW = np.zeros((nclasses, nfeatures, nfeatures)), np.zeros((nclasses, nfeatures, nfeatures)), np.zeros((nclasses, nfeatures, nfeatures))

W = np.zeros((nclasses, nfeatures, 1))

W0 = np.zeros((nclasses))
# Calculating SW, W & W0 #

for i in range(nclasses):

    class_means[i] = np.mean(im_train[class_indexes[i]], axis=0)

    other_class.append(np.delete(im_train, class_indexes[i], axis=0)) # one-versus-the-rest approach

    other_class_means[i] = np.mean(other_class[i], axis=0)

    between_class1 = np.subtract(im_train[class_indexes[i]].reshape(-1, nfeatures), 

                                 class_means[i])

    SW_one[i] = between_class1.T.dot(between_class1)

    between_class2 = np.subtract(other_class[i], other_class_means[i])

    SW_two[i] = between_class2.T.dot(between_class2)

    SW[i] = SW_one[i] + SW_two[i]

    W[i] = np.dot(np.linalg.pinv(SW[i]), 

                  np.subtract(other_class_means[i], 

                              class_means[i]).reshape(-1, 1))

    W0[i] = -0.5 * np.dot(W[i].T, (class_means[i] + other_class_means[i]))

print(SW.shape)

print(W.shape)

print(W0.shape)
im_test = np.loadtxt('../input/test.csv', delimiter=',', dtype=int, skiprows=1)

im_test.shape
Y = np.zeros((len(im_test), nclasses))

predict = np.zeros((len(im_test)), dtype=int)

for j in range(len(im_test)):

    for i in range(nclasses):

        Y[j, i] = np.dot(W[i].T,  im_test[j]) + W0[i]

    predict[j] = np.argmin(Y[j])
predict[:10]
for i in range(10):

    plt.subplot(1, 10, i+1) # plot index can not be 0

    plt.imshow(im_test[i].reshape(28, 28))

    plt.axis('off')

plt.show()
submission = pd.DataFrame({"ImageId": np.arange(1, len(im_test)+1), "Label": predict})

submission.to_csv('submission.csv', index=False)