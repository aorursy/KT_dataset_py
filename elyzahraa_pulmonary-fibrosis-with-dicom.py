import os

import numpy as np

import pandas as pd

import seaborn as sns

import re

import cv2

import matplotlib.pyplot as plt

from matplotlib import rc



import pydicom

import plotly.graph_objs as go

import glob as glob

import imageio



# Segmentation

from skimage import morphology

from scipy import ndimage

from skimage import measure

from skimage.transform import resize

from sklearn.cluster import KMeans

from plotly.tools import FigureFactory as FF



import warnings

warnings.simplefilter(action='ignore')

sns.set(style="white")



%matplotlib inline

train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

sample = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
train.head()
train.tail()
train.info()
train.isnull().sum()


print(f"No of patients are {train['Patient'].count()} with {train['Patient'].value_counts().shape[0]} unique patient ID")
train.shape
columns = list(train.columns)

print(f'The colums are {columns}')
f, ax = plt.subplots(figsize=(7, 5))

sns.despine(f)

sns.distplot(train['Age'])

plt.title("PATIENT AGE DISTRIBUTION")
plt.figure(figsize=(8, 4))

sns.countplot(x='SmokingStatus', data=train)

plt.title("Smoking Status Patient")

plt.show()
sns.scatterplot(x='FVC', y='Percent', hue='Sex', data=train, palette='PuOr')

plt.title('Correlation between FVC and Percent')

# FVC vs AGE

sns.scatterplot(x='Age', y='FVC', hue='SmokingStatus', data=train, palette='Set1')

plt.title('Correlation between FVC and Age')
plt.title('Correlation between FVC and Age')

sns.kdeplot(train['FVC'], train['Age'], cmap="Blues", shade=True, shade_lowest=True )

heat_map = sns.heatmap(train.corr(method='pearson'), annot=True, fmt='.2f', linewidths=2)

heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45);
#Image pacient on Data



dir = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/20.dcm"

data = pydicom.dcmread(dir)



print("Patient id.......:", data.PatientID, "\n" +

      "Modality.........:", data.Modality, "\n" +

      "Rows.............:", data.Rows, "\n" +

      "Columns..........:", data.Columns)



plt.figure(figsize = (5, 5))

plt.imshow(data.pixel_array, cmap="Blues_r")

plt.axis('off');
dir = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430"

data = []





files = []

for dcm in list(os.listdir(dir)):

    files.append(dcm) 

files.sort(key=lambda f: int(re.sub('\D', '', f)))





for dcm in files:

    path = dir + "/" + dcm

    data.append(pydicom.dcmread(path))





fig=plt.figure(figsize=(18, 6))

columns = 10

rows = 3



for i in range(1, columns*rows +1):

    img = data[i-1].pixel_array

    fig.add_subplot(rows, columns, i)

    plt.imshow(img, cmap="Blues_r")

    plt.title(i, fontsize = 12)

    plt.axis('off');



dir = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/18.dcm"



data = pydicom.dcmread(dir)

print(data)

image = data.pixel_array
def transform_to_hu(data, image):

    intercept = data.RescaleIntercept

    slope = data.RescaleSlope

    hu_image = image * slope + intercept



    return hu_image
def window_image(image, window_center, window_width):

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    window_image = image.copy()

    window_image[window_image < img_min] = img_min

    window_image[window_image > img_max] = img_max

    

    return window_image
dir = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/16.dcm"

data = pydicom.dcmread(dir)



def remove_noise(dir, display=False):

    data = pydicom.dcmread(dir)

    image = data.pixel_array

    

    hu_image = transform_to_hu(data, image)

    brain_image = window_image(hu_image, 40, 80)



    # morphology.dilation creates a segmentation of the image

    # If one pixel is between the origin and the edge of a square of size

    # 5x5, the pixel belongs to the same class

    

    # We can instead use a circule using: morphology.disk(2)

    # In this case the pixel belongs to the same class if it's between the origin

    # and the radius

    

    segmentation = morphology.dilation(brain_image, np.ones((5, 5)))

    labels, label_nb = ndimage.label(segmentation)

    

    label_count = np.bincount(labels.ravel().astype(np.int))

    # The size of label_count is the number of classes/segmentations found

    

    # We don't use the first class since it's the background

    label_count[0] = 0

    

    # We create a mask with the class with more pixels

    # In this case should be the brain

    mask = labels == label_count.argmax()

    

    # Improve the brain mask

    mask = morphology.dilation(mask, np.ones((5, 5)))

    mask = ndimage.morphology.binary_fill_holes(mask)

    mask = morphology.dilation(mask, np.ones((3, 3)))

    

    # Since the the pixels in the mask are zero's and one's

    # We can multiple the original image to only keep the brain region

    masked_image = mask * brain_image



    if display:

        plt.figure(figsize=(15, 2.5))

        plt.subplot(141)

        plt.imshow(brain_image)

        plt.title('Original Image')

        plt.axis('off')

        

        plt.subplot(142)

        plt.imshow(mask)

        plt.title('Mask')

        plt.axis('off')



        plt.subplot(143)

        plt.imshow(masked_image)

        plt.title('Final Image')

        plt.axis('off')

    

    return masked_image
_ = remove_noise(dir, display=True)
dir = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/30.dcm"

data = pydicom.dcmread(dir)



def crop_image(image, display=False):

    # Create a mask with the background pixels

    mask = image == 0



    # Find the brain area

    coords = np.array(np.nonzero(~mask))

    top_left = np.min(coords, axis=1)

    bottom_right = np.max(coords, axis=1)

    

    # Remove the background

    croped_image = image[top_left[0]:bottom_right[0],

                top_left[1]:bottom_right[1]]

    

    return croped_image

final = remove_noise(dir)
plt.imshow(final)
final = crop_image(final)
plt.imshow(final)