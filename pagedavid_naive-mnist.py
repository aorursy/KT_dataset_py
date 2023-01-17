import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import scipy.ndimage
from scipy.io import loadmat

mnist = loadmat("../input/mnist-original/mnist-original.mat")

mnist_data = mnist["data"].T

mnist_label = mnist["label"][0]
p = np.random.permutation(len(mnist_data))

mnist_data = mnist_data[p] / 255.

mnist_label = mnist_label[p].astype(int)
def show_image(index):

    first_image = mnist_data[index]

    first_image = np.array(first_image, dtype='float')

    first_image = first_image.reshape(28, 28)

    plt.imshow(first_image, cmap='gray')

    plt.show()

    print(mnist_label[index])

show_image(807)
mnist_train_data = mnist_data[:60000]

mnist_train_label = mnist_label[:60000]

mnist_test_data = mnist_data[60000:]

mnist_test_label = mnist_label[60000:]
mask = np.zeros((10, 784))

for i in range(60000):

    mask[mnist_train_label[i]] += mnist_train_data[i]

mask = mask / 60000.
pixels = np.array(mask[7], dtype='float')

pixels = pixels.reshape(28, 28)

plt.imshow(pixels, cmap='gray')

plt.show()
def cal_mask_dis(img, test_mask):

    res = list()

    for i in range(10):

        res.append(np.linalg.norm(test_mask[i] - img, ord=2))

    return res.index(min(res))
correct = 0

for i in range(len(mnist_test_label)):

    if cal_mask_dis(mnist_test_data[i], mask) == mnist_test_label[i]:

        correct += 1

correct / len(mnist_test_label)