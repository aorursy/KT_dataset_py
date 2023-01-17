# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random

image_dir = '/kaggle/input/face-images-with-marked-landmark-points/face_images.npz'
keypoints_dir = '/kaggle/input/face-images-with-marked-landmark-points/face_images.npz'

# Load the dataset 
images = np.load(image_dir)['face_images']
facial_keypoints = pd.read_csv(keypoints_dir)

# Standardize the values of images 
images = images/255
# Plot a random image
m = images.shape[2] # Number of samples
# Plot a random image
index = random.randint(0, m)
plt.imshow(images[:,:,index], cmap='gray')
