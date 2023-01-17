import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import dicom

import cv2

import os

import matplotlib.pyplot as plt



%config InlineBackend.figure_format = 'retina'
def show(slice):

    plt.imshow(slice, cmap=plt.cm.bone)
os.chdir("../input/sample_images")
files = os.listdir()
os.chdir('0d941a3ad6c889ac451caf89c46cb92a')
files = os.listdir()

files





imgs = []

for i in files:

    try:

        ds = dicom.read_file(i, force = True)

        imgs.append(ds)

    except:

        pass
#borrowed from r4m0n:

imgs.sort(key = lambda x: int(x.InstanceNumber))

full_img = np.stack([s.pixel_array for s in imgs])
full_img[0].shape
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)





for i in range(36):

    plt.subplot(6,6,i+1)

    show(full_img[4*i,:,:])    

    plt.xticks([])

    plt.yticks([])
for i in range(36):

    plt.subplot(6,6,i+1)

    img = cv2.resize(full_img[:,20+ 12*i,:], (256, 256))

    show(img)    

    plt.xticks([])

    plt.yticks([])
for i in range(36):

    plt.subplot(6,6,i+1)

    img = cv2.resize(full_img[:,:,20+ 12*i], (256, 256))

    show(img)    

    plt.xticks([])

    plt.yticks([])