import cv2

import numpy as np

from glob import glob

from pathlib import Path

from scipy import ndimage

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from skimage.color import rgb2gray

from skimage.io import imread, imsave

from scipy.ndimage import binary_fill_holes

from sklearn.metrics import confusion_matrix

from skimage.measure import label, regionprops

from skimage.filters import sobel, gaussian

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
imagem = rgb2gray(imread('../input/ultrasound-original/cropped/1100208028.bmp'))
n_clusters = 2



filterGaussian = gaussian(imagem, sigma=3)

    

X = filterGaussian.reshape((-1, 1))  

k_means = KMeans(n_clusters).fit(X)

imgKmeans = k_means.labels_

kmeans = imgKmeans.reshape(filterGaussian.shape)

    

    

plt.imshow(kmeans,cmap='gray')
region = np.zeros(kmeans.shape)

label_image = label(kmeans, connectivity=1)

num_labels = np.unique(label_image)



for indice in num_labels:



    props = regionprops(label_image)



    area = [reg.area for reg in props]

    largest_label_ind = np.argmax(area)

    largest_label = props[largest_label_ind].label



    region[label_image == largest_label] = indice

    

plt.imshow(region, cmap='gray')
def border(imagem):

    imagem[:1,:] = 0.0

    imagem[496:,:] = 0.0

    imagem[:,:1] = 0.0

    imagem[:,322:] = 0.0



    return imagem
region = border(region)

plt.imshow(region, cmap='gray')
bordaSo = sobel(region)

plt.imshow(bordaSo, cmap='gray')
fill_holes = binary_fill_holes(bordaSo)

plt.imshow(fill_holes, cmap='gray')
fill_holes.shape
multi = fill_holes * imagem

fig, aux = plt.subplots(1,3, figsize=(30,10))

aux[0].imshow(imagem, cmap='gray')

aux[1].imshow(fill_holes, cmap='gray')

aux[2].imshow(multi, cmap='gray')