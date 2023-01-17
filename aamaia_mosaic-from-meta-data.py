import matplotlib.pyplot as plt

import cv2

import numpy as np



def load_img(image_name, s):

    if 'train' in image_name:

        path = '../input/train-jpg/{}'.format(image_name)

    else:

        path = '../input/test-jpg/{}'.format(image_name)

    bgr = cv2.imread(path)

    if bgr is not None:

        img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (s, s))  

        return img

    else:

        return np.zeros((s, s, 3))





mosaic_idx = np.array([

    [None, 'test_16387.jpg', 'test_32642.jpg', 'train_29248.jpg', 'test_1707.jpg'],

    [None, 'train_26144.jpg', 'train_39316.jpg', 'train_17700.jpg', None],

    ['train_22759.jpg', 'train_25804.jpg', None, None, None],

    

])



s = 50

mosaic = np.zeros((s*3, s*5, 3))



for j in range(3):

    for i in range(5):

        if mosaic_idx[j, i]:

            mosaic[j*s:s*(j+1), i*s:(i+1)*s, :] = load_img(mosaic_idx[j, i], s)    



plt.imshow(mosaic)
