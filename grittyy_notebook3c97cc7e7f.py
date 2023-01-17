import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import dicom

import cv2

from skimage import data, io, filters

import os

import matplotlib.pyplot as plt

import pylab



from mpl_toolkits.mplot3d.art3d import Poly3DCollection



%config InlineBackend.figure_format = 'retina'
def show(slice):

    plt.imshow(slice, cmap=plt.cm.bone)
sample_images = "../input/sample_images"

files = os.listdir(sample_images)

labels = pd.read_csv("../input/stage1_labels.csv")

cancers = labels[labels['cancer'] > 0]

cancer_samples = np.intersect1d(files, cancers['id'])

cancer_samples
cid = cancer_samples[0]

cid = files[9]
cimages = os.listdir(os.path.join(sample_images, cid))
imgs = []

for i in cimages:

        ds = dicom.read_file(os.path.join(sample_images, cid, i))

        imgs.append(ds)
len(imgs)
#sorting based on InstanceNumber stolen from r4m0n's script: 

imgs.sort(key = lambda x: int(x.InstanceNumber))

full_img = np.stack([s.pixel_array for s in imgs])
full_img.shape
%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (7.0, 17.0)



for i in range(96):

    plt.subplot(16,6,i+1)

    show(full_img[int(full_img.shape[0]/96*i),:,:])    

    plt.xticks([])

    plt.yticks([])
for i in range(96):

    plt.subplot(16,6,i+1)

    img = cv2.resize(full_img[:,50+ 4*i,:], (256, 256))

    show(img)    

    plt.xticks([])

    plt.yticks([])
for i in range(96):

    plt.subplot(16,6,i+1)

    img = cv2.resize(full_img[:,:,20+ 5*i], (256, 256))

    show(img)    

    plt.xticks([])

    plt.yticks([])