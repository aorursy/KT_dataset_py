import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.io as sio

from sklearn.preprocessing import StandardScaler

from PIL import Image





def plot_images(X_image, num_images, title):

    num_images_x = int(np.sqrt(num_images))

    fig3, ax_array = plt.subplots(num_images_x, num_images_x, figsize=(8, 8))

    for i, ax in enumerate(ax_array.flat):

        image = X_image[i, :]

        num_rows = num_cols = int(np.sqrt(len(image)))

        ax.imshow(image.reshape(num_rows, num_cols, order='F'), cmap='gray')

        fig3.suptitle(title)

        ax.axis('off')



os.chdir('/kaggle/input/coursera-andrewng-ml-dataset')

data2 = sio.loadmat('ex7faces.mat')

X_image = data2['X']

num_images = 100

plot_images(X_image, num_images, 'Original face images')
from sklearn.decomposition import PCA

scaler = StandardScaler()

X_image_scaled = scaler.fit_transform(X_image)

pca = PCA(n_components=100)

pca.fit(X_image_scaled)
X_image_reduced = pca.transform(X_image_scaled)

print(X_image_reduced[0, :])
X_image_recovered = pca.inverse_transform(X_image_reduced)

plot_images(X_image_recovered, num_images, 'Recovered face images')
pca = PCA(n_components=36)

pca.fit(X_image_scaled)

X_image_reduced = pca.transform(X_image_scaled)

X_image_recovered = pca.inverse_transform(X_image_reduced)

plot_images(X_image_recovered, num_images, 'Recovered face images from 36 principal components')