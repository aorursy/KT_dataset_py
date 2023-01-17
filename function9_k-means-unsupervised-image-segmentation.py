import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import numpy as np

from sklearn.cluster import KMeans

from PIL import Image

from skimage.color import rgb2hsv

import os
os.listdir('../input/tray-food-segmentation/TrayDataset/TrayDataset/XTest')
img = Image.open(r'../input/tray-food-segmentation/TrayDataset/TrayDataset/XTest/8006a.jpg')

x = np.array(img)

imshow(x)
z = np.dstack((x,rgb2hsv(x)))

z.shape
vectorized = np.float32(z.reshape((-1,6)))

vectorized.shape
kmeans = KMeans(random_state=0, init='random', n_clusters=8)

labels = kmeans.fit_predict(vectorized)
labels.shape
pic = labels.reshape(256,416)
f, axarr = plt.subplots(1,2,figsize=(15,15))

axarr[0].set_xlabel('Original Image', fontsize=12)

axarr[1].set_xlabel('Segmented Image', fontsize=12)  

axarr[0].imshow(x)

axarr[1].imshow(pic)
img = Image.open(r'../input/tray-food-segmentation/TrayDataset/TrayDataset/XTest/2002a.jpg')

x = np.array(img)

z = np.dstack((x,rgb2hsv(x)))

z.shape

vectorized = np.float32(z.reshape((-1,6)))

kmeans = KMeans(random_state=32, init='random', n_clusters=7)

james = kmeans.fit_predict(vectorized)

pic = james.reshape(256,416)

f, axarr = plt.subplots(1,2,figsize=(15,15))

axarr[0].set_xlabel('Original Image', fontsize=12)

axarr[1].set_xlabel('Segmented Image', fontsize=12)  

axarr[0].imshow(x)

axarr[1].imshow(pic)
img = Image.open(r'../input/tray-food-segmentation/TrayDataset/TrayDataset/XTest/1005a.jpg')

x = np.array(img)

z = np.dstack((x,rgb2hsv(x)))

z.shape

vectorized = np.float32(z.reshape((-1,6)))

kmeans = KMeans(random_state=32, init='random', n_clusters=6)

james = kmeans.fit_predict(vectorized)

pic = james.reshape(256,416)

f, axarr = plt.subplots(1,2,figsize=(15,15))

axarr[0].set_xlabel('Original Image', fontsize=12)

axarr[1].set_xlabel('Segmented Image', fontsize=12)  

axarr[0].imshow(x)

axarr[1].imshow(pic)