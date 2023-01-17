# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
img = load_sample_image('flower.jpg')
plt.imshow(img)
img = img/255
X = img.reshape((-1, 3)).copy()
from sklearn.cluster import KMeans, MiniBatchKMeans
kmeans = KMeans(n_clusters=32)
kmeans.fit(X)
X.shape
mnkmeans = MiniBatchKMeans(n_clusters=32)
mnkmeans.fit(X)
new_img = mnkmeans.cluster_centers_[mnkmeans.labels_]
new_img = new_img.reshape((427, 640, 3))
new_img.shape
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img)
axes[1].imshow(img)
kmn = MiniBatchKMeans(n_clusters=64)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)
kmn = MiniBatchKMeans(n_clusters=16)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)
kmn = MiniBatchKMeans(n_clusters=8)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)
kmn = MiniBatchKMeans(n_clusters=4)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)
kmn = MiniBatchKMeans(n_clusters=256)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)
kmn = MiniBatchKMeans(n_clusters=2)
kmn.fit(img.reshape(-1, 3))
new_img = kmn.cluster_centers_[kmn.labels_]
fg, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_img.reshape(427, 640 ,3))
axes[1].imshow(img)
