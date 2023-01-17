# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np  # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sc

import os

import dicom

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

%matplotlib inline
# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.

data_folder='../input/'

sample_folder=data_folder+'sample_images/'

from subprocess import check_output

print(check_output(["ls", "-l", data_folder]).decode("utf8"))



patients=os.listdir(sample_folder)

patients.sort()

print(patients)
def load_scan(path):

    slices=[dicom.read_file(path+'/'+s) for s in os.listdir(path)]

    slices.sort(key=lambda x: int(x.InstanceNumber))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:

        s.SliceThickness = slice_thickness

    return slices



def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans]).astype(np.int16)



    # Set outside-of-scan pixels to 0; the intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
def load_and_process_scans(filename):

    scans = load_scan(filename)

    images = get_pixels_hu(scans)

    print("Loaded {} scans/images from {}".format(len(scans),filename))

    return scans,images

scans,images = load_and_process_scans(sample_folder+patients[8])
def plot_slices(slices):

    fig = plt.figure(figsize=(8,8))

    plot_ids=[221,222,223,224]

    for i,plot_id in enumerate(plot_ids):

        ax  = fig.add_subplot(plot_id)

        _ = ax.imshow(images[slices[i]], cmap='gray')

        ax.set_title("Slice #{}".format(slices[i]))
fractions = np.linspace(0.0,1.0,6)[1:-1]

plot_slices(slices = (fractions*(len(scans)-1)).astype(int))
fractions = np.linspace(0.4,0.6,4); print(fractions)

plot_slices(slices = (fractions*(len(scans)-1)).astype(int))
def find_threshold(image):

    kmeans = KMeans(n_clusters=2).fit(image.reshape(-1,1))

    c = kmeans.cluster_centers_

    return 0.5*(c[0]+c[1])



def plot_slices_hist(slices):

    fig = plt.figure(figsize=(8,8))

    plot_ids=[221,222,223,224]

    max_count = np.max([np.max(np.histogram(images[s],bins=80)[0]) for s in slices])  # for scaling

    for i,plot_id in enumerate(plot_ids):

        ax  = fig.add_subplot(plot_id)

        threshold = find_threshold(images[slices[i]])

        _ = ax.hist(images[slices[i]].flatten(), bins=40)

        _ = ax.axvline(threshold, color='0.8', ls='dashed', lw=1)

        ax.set_ylim(0,max_count)

        ax.set_title("Slice #{}".format(slices[i]))
plot_slices_hist(slices = (fractions*(len(scans)-1)).astype(int))
s = 65

threshold = find_threshold(images[s])

binary_image = (images[s] > threshold).astype(int)

_ = plt.imshow(binary_image,cmap='gray')
from skimage import morphology

from skimage import measure

image_o = morphology.binary_opening(binary_image,morphology.diamond(7))

_ = plt.imshow(image_o,cmap='gray')
labels = measure.label(image_o+1,neighbors=4)  # add one to image_o because '0' is considered to be 'background'

label_vals = np.unique(labels); print(label_vals)
fig = plt.figure(figsize=(7,7))

ax0 = fig.add_subplot(221); ax0.imshow(labels); ax0.set_title('Segmentation')

ax1 = fig.add_subplot(222); ax1.imshow(labels==2, cmap='gray'); ax1.set_title('Wall')

ax2 = fig.add_subplot(223); ax2.imshow(labels==3, cmap='gray'); ax2.set_title('Left')

ax3 = fig.add_subplot(224); ax3.imshow(labels==4, cmap='gray'); ax3.set_title('Right')

print("Area of wall       = {} pixels".format(np.sum(labels==2)))

print("Area of left  lung = {} pixels".format(np.sum(labels==3)))

print("Area of right lung = {} pixels".format(np.sum(labels==4)))
from skimage import feature

edges_left  = feature.canny(labels==3,sigma=4)

edges_right = feature.canny(labels==4,sigma=4)

fig = plt.figure(figsize=(12,7))

ax1 = fig.add_subplot(121); ax1.imshow(edges_left, cmap='gray'); ax1.set_title('Left Edges')

ax2 = fig.add_subplot(122); ax2.imshow(edges_right, cmap='gray'); ax2.set_title('Right Edges')