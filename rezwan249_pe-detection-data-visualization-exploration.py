# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import glob



import os

import numpy as np

import pandas as pd

import pydicom as dcm

import matplotlib

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

import glob

# import gdcm

from matplotlib import animation, rc

from plotly.subplots import make_subplots

import plotly.graph_objs as go



import pydicom

import scipy.ndimage

# import gdcm

import imageio





import os

import copy

from datetime import timedelta, datetime

import imageio

import matplotlib.pyplot as plt

from matplotlib import cm

import multiprocessing

import numpy as np

import os

from pathlib import Path

import pydicom

import pytest

import scipy.ndimage as ndimage

from scipy.ndimage.interpolation import zoom

from skimage import measure, morphology, segmentation

from skimage.transform import resize

from time import time, sleep

from tqdm import trange, tqdm

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import *

from tensorflow.data import Dataset

import torch

from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

import warnings

import seaborn as sns

import glob as glob

import imageio

from IPython.display import Image



#for masking

from skimage.measure import label,regionprops

from sklearn.cluster import KMeans

from skimage.segmentation import clear_border



import onnx

# +++++++++++++++?



root = "/kaggle/input/rsna-str-pulmonary-embolism-detection/"



for item in os.listdir(root):

    path = os.path.join(root, item)

    if os.path.isfile(path):

        print(path)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv")

train_data.head()
train_data.shape
train_data.columns
train_data.info()
train_data.describe()
pd.isnull(train_data).any()
def rhead(x, nrow = 6, ncol = 4):

    pd.set_option('display.expand_frame_repr', False)

    seq = np.arange(0, len(x.columns), ncol)

    for i in seq:

        print(x.loc[range(0, nrow), x.columns[range(i, min(i+ncol, len(x.columns)))]])

    pd.set_option('display.expand_frame_repr', True)
rhead(train_data)
for i in range(train_data.shape[1]-3):



    train_data.hist(column=train_data.columns[i+3])
import matplotlib.pyplot as plt

import plotly.express as px



train_data_drop = train_data.drop(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], axis=1)


train_data_select = train_data_drop.sum(axis=0).sort_values().reset_index()

train_data_select.columns = ['columns', 'nonzero_records']



fig = px.bar(

    train_data_select, 

    x='nonzero_records', 

    y='columns', 

    orientation='h', 

    title='Columns and non zero samples', 

    height=800, 

    width=600

)

fig.show()


train_data_select = train_data_drop.astype(bool).sum(axis=1).reset_index()

train_data_select.columns = ['rows', 'count']



train_data_select = train_data_select.groupby(['count'])['rows'].count().reset_index()

fig = px.pie(

    train_data_select, 

    values=round((100 * train_data_select['rows'] / len(train_data)), 2), 

    names="count", 

    title='Every sample (Percent)', 

    width=500, 

    height=500

)

fig.show()


f = plt.figure(figsize=(20, 20))

plt.matshow(train_data_drop.corr(), fignum=f.number)

plt.xticks(range(train_data_drop.shape[1]), train_data_drop.columns, fontsize=12, rotation=90)

plt.yticks(range(train_data_drop.shape[1]), train_data_drop.columns, fontsize=12)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=13)
train_data_dir = "../input/rsna-str-pulmonary-embolism-detection/train"

print('Total number of dirictories in training set {}'.format(len(os.listdir(train_data_dir))))
# !pip install dicom
import vtk

from vtk.util import numpy_support

import cv2



reader = vtk.vtkDICOMImageReader()

def get_img(path):

    reader.SetFileName(path)

    reader.Update()

    _extent = reader.GetDataExtent()

    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]



    ConstPixelSpacing = reader.GetPixelSpacing()

    imageData = reader.GetOutput()

    pointData = imageData.GetPointData()

    arrayData = pointData.GetArray(0)

    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)

    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

    ArrayDicom = cv2.resize(ArrayDicom,(512,512))

    return ArrayDicom


import matplotlib.pyplot as plt





def show_dicom_images(dcom):

    f, ax = plt.subplots(1,1, figsize=(12,10))

    ax.imshow(dcom, cmap=plt.cm.bone)

    ax.axis('off')

    ax.set_title('Original DICOM Image')

    plt.show()

    

#test read a dcom file and view it

img_path = "../input/rsna-str-pulmonary-embolism-detection/train/005a0dbcb4b7/4ceaee66edc8/01a737504be7.dcm"

img_get = get_img(img_path)

show_dicom_images(img_get)
import pydicom as dcm

fig, ax = plt.subplots(figsize=(12, 12))

ax.imshow(dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/00c07cd8129d/8877e4d12ce9/00feb47a8d76.dcm").pixel_array)

train_one_img = dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/00c07cd8129d/8877e4d12ce9/00feb47a8d76.dcm").pixel_array

print(train_one_img.shape)
train_one_img_info = dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/00c07cd8129d/8877e4d12ce9/00feb47a8d76.dcm")

print(train_one_img_info)
data_path = '../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/'



output_path = '../input/output/'



train_image_files = sorted(glob.glob(os.path.join(data_path, '*','*.dcm')))

train_image_files_list = os.listdir(data_path)



train_image_files_list.sort()



print('Some sample ID''s :', len(train_image_files))

print("\n".join(train_image_files[:5]))
def load_scan(path):

    """

    Loads scans from a folder and into a list.

    

    Parameters: path (Folder path)

    

    Returns: slices (List of slices)

    """

    

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices
def get_pixels_hu(scans):

    """

    Converts raw images to Hounsfield Units (HU).

    

    Parameters: scans (Raw images)

    

    Returns: image (NumPy array)

    """

    

    image = np.stack([s.pixel_array for s in scans])

    image = image.astype(np.int16)



    # Since the scanning equipment is cylindrical in nature and image output is square,

    # we set the out-of-scan pixels to 0

    image[image == -2000] = 0

    

    

    # HU = m*P + b

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
train_img_scans = load_scan(data_path + train_image_files_list[0])

train_images = get_pixels_hu(train_img_scans)



#We'll be taking a random slice to perform segmentation:



for imgs in range(len(train_images[0:5])):

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))

    ax1.imshow(train_images[imgs], cmap=plt.cm.bone)

    ax1.set_title("Original Slice")

    

    ax2.imshow(train_images[imgs], cmap=plt.cm.bone)

    ax2.set_title("Original Slice")

    

    ax3.imshow(train_images[imgs], cmap=plt.cm.bone)

    ax3.set_title("Original Slice")

    plt.show()
from IPython.display import Image



def set_lungwin(img, hu=[-1200., 600.]):

    lungwin = np.array(hu)

    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])

    newimg[newimg < 0] = 0

    newimg[newimg > 1] = 1

    newimg = (newimg * 255).astype('uint8')

    return newimg





scans = load_scan(data_path + train_image_files_list[0])

scan_array = set_lungwin(get_pixels_hu(scans))



imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)

Image(filename="/tmp/gif.gif", format='png')
def generate_markers(image):

    """

    Generates markers for a given image.

    

    Parameters: image

    

    Returns: Internal Marker, External Marker, Watershed Marker

    """

    

    #Creation of the internal Marker

    marker_internal = image < -400

    marker_internal = segmentation.clear_border(marker_internal)

    marker_internal_labels = measure.label(marker_internal)

    

    areas = [r.area for r in measure.regionprops(marker_internal_labels)]

    areas.sort()

    

    if len(areas) > 2:

        for region in measure.regionprops(marker_internal_labels):

            if region.area < areas[-2]:

                for coordinates in region.coords:                

                       marker_internal_labels[coordinates[0], coordinates[1]] = 0

    

    marker_internal = marker_internal_labels > 0

    

    # Creation of the External Marker

    external_a = ndimage.binary_dilation(marker_internal, iterations=10)

    external_b = ndimage.binary_dilation(marker_internal, iterations=55)

    marker_external = external_b ^ external_a

    

    # Creation of the Watershed Marker

    marker_watershed = np.zeros((512, 512), dtype=np.int)

    marker_watershed += marker_internal * 255

    marker_watershed += marker_external * 128

    

    return marker_internal, marker_external, marker_watershed
train_img_scans = load_scan(data_path + train_image_files_list[0])

train_images = get_pixels_hu(train_img_scans)

print(len(train_img_scans))




test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(train_images[2])



f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,15))



ax1.imshow(test_patient_internal, cmap='gray')

ax1.set_title("Internal Marker")

ax1.axis('off')



ax2.imshow(test_patient_external, cmap='gray')

ax2.set_title("External Marker")

ax2.axis('off')



ax3.imshow(test_patient_watershed, cmap='gray')

ax3.set_title("Watershed Marker")

ax3.axis('off')



plt.show()
sample_image = pydicom.dcmread(train_image_files[7])

img = sample_image.pixel_array



plt.imshow(img, cmap='gray')

plt.title('Original Image')
img = (img + sample_image.RescaleIntercept) / sample_image.RescaleSlope

img = img < -400 #HU unit range for lungs CT SCAN



plt.imshow(img, cmap='gray')

plt.title('Binary Mask Image')
img = clear_border(img)

plt.imshow(img, cmap='gray')

plt.title('Cleaned Border Image')
img = label(img)

plt.imshow(img, cmap='gray')
areas = [r.area for r in regionprops(img)]

areas.sort()

if len(areas) > 2:

    for region in regionprops(img):

        if region.area < areas[-2]:

            for coordinates in region.coords:                

                img[coordinates[0], coordinates[1]] = 0

img = img > 0

plt.imshow(img, cmap='gray')


def make_pemask(img, display=False):

    row_size= img.shape[0]

    col_size = img.shape[1]

    

    mean = np.mean(img)

    std = np.std(img)

    img = img-mean

    img = img/std

    

    # Find the average pixel value near the lungs

        # to renormalize washed out images

    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 

    mean = np.mean(middle)  

    max = np.max(img)

    min = np.min(img)

    

    # To improve threshold finding, I'm moving the 

    # underflow and overflow on the pixel spectrum

    img[img==max]=mean

    img[img==min]=mean

    

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))

    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold = np.mean(centers)

    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image



    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  

    # We don't want to accidentally clip the lung.



    eroded = morphology.erosion(thresh_img,np.ones([3,3]))

    dilation = morphology.dilation(eroded,np.ones([8,8]))



    labels = measure.label(dilation) # Different labels are displayed in different colors

    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)

    good_labels = []

    for prop in regions:

        B = prop.bbox

        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:

            good_labels.append(prop.label)

    mask = np.ndarray([row_size,col_size],dtype=np.int8)

    mask[:] = 0





    #  After just the lungs are left, we do another large dilation

    #  in order to fill in and out the lung mask 

    

    for N in good_labels:

        mask = mask + np.where(labels==N,1,0)

    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation



    if (display):

        fig, ax = plt.subplots(3, 2, figsize=[12, 12])

        ax[0, 0].set_title("Original")

        ax[0, 0].imshow(img, cmap='gray')

        ax[0, 0].axis('off')

        ax[0, 1].set_title("Threshold")

        ax[0, 1].imshow(thresh_img, cmap='gray')

        ax[0, 1].axis('off')

        ax[1, 0].set_title("After Erosion and Dilation")

        ax[1, 0].imshow(dilation, cmap='gray')

        ax[1, 0].axis('off')

        ax[1, 1].set_title("Color Labels")

        ax[1, 1].imshow(labels)

        ax[1, 1].axis('off')

        ax[2, 0].set_title("Final Mask")

        ax[2, 0].imshow(mask, cmap='gray')

        ax[2, 0].axis('off')

        ax[2, 1].set_title("Apply Mask on Original")

        ax[2, 1].imshow(mask*img, cmap='gray')

        ax[2, 1].axis('off')

        

        plt.show()

    return mask*img
# Select a sample

path = "../input/rsna-str-pulmonary-embolism-detection/train/000f7f114264/9f7378c3b2ab/0003aa3e734b.dcm"

dataset = pydicom.dcmread(path)

img = dataset.pixel_array



# Masked image

mask_img = make_pemask(img, display=True)
import re

train_img_dir = "../input/rsna-str-pulmonary-embolism-detection/train/000f7f114264/9f7378c3b2ab"

datasets = []



# First Order the files in the dataset

files = []

for dcm in list(os.listdir(train_img_dir)):

    files.append(dcm) 

files.sort(key=lambda f: int(re.sub('\D', '', f)))



# Read in the Dataset

for dcm in files:

    path = train_img_dir + "/" + dcm

    datasets.append(pydicom.dcmread(path))

    

imgs = []

for data in datasets:

    img = data.pixel_array

    imgs.append(img)

    

    

# Show masks

fig=plt.figure(figsize=(16, 6))

columns = 10

rows = 3



for i in range(1, columns*rows +1):

    img = make_pemask(datasets[i-1].pixel_array)

    fig.add_subplot(rows, columns, i)

    plt.imshow(img, cmap="gray")

    plt.title(i, fontsize = 9)

    plt.axis('off');