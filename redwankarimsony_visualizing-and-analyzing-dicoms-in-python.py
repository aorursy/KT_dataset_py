!pip install  chart_studio

# common packages 

import numpy as np 

import os

import copy

from math import *

import matplotlib.pyplot as plt

from functools import reduce

from glob import glob



# reading in dicom files

import pydicom



# skimage image processing packages

from skimage import measure, morphology

from skimage.morphology import ball, binary_closing

from skimage.measure import label, regionprops



# scipy linear algebra functions 

from scipy.linalg import norm

import scipy.ndimage



# ipywidgets for some interactive plots

from ipywidgets.widgets import * 

import ipywidgets as widgets



# plotly 3D interactive graphs 

import plotly

from plotly.graph_objs import *

import chart_studio

chart_studio.tools.set_credentials_file(username='redwankarimsony', api_key='aEbXWsleQv7PJrAtOkBk')

# set plotly credentials here 

# this allows you to send results to your account plotly.tools.set_credentials_file(username=your_username, api_key=your_key)
patient_id = '6897fa9de148'

patient_folder = f'../input/rsna-str-pulmonary-embolism-detection/train/{patient_id}/'

data_paths = glob(patient_folder + '/*/*.dcm')



# Print out the first 5 file names to verify we're in the right folder.

print (f'Total of {len(data_paths)} DICOM images.\nFirst 5 filenames:' )

data_paths[:5]
def load_scan(paths):

    slices = [pydicom.read_file(path ) for path in paths]

    slices.sort(key = lambda x: int(x.InstanceNumber), reverse = True)

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
# set path and load files 

patient_dicom = load_scan(data_paths)

patient_pixels = get_pixels_hu(patient_dicom)

#sanity check

plt.imshow(patient_pixels[80], cmap=plt.cm.bone)
def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]

    vals = vals[vals != bg]

    if len(counts) > 0:

        return vals[np.argmax(counts)]

    else:

        return None

    

def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2. 

    # 0 is treated as background, which we do not want

    binary_image = np.array(image >= -700, dtype=np.int8)+1

    labels = measure.label(binary_image)

 

    # Pick the pixel in the very corner to determine which label is air.

    # Improvement: Pick multiple background labels from around the  patient

    # More resistant to “trays” on which the patient lays cutting the air around the person in half

    background_label = labels[0,0,0]

 

    # Fill the air around the person

    binary_image[background_label == labels] = 2

 

    # Method of filling the lung structures (that is superior to 

    # something like morphological closing)

    if fill_lung_structures:

        # For every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

 

            if l_max is not None: #This slice contains some lung

                binary_image[i][labeling != l_max] = 1

    binary_image -= 1 #Make the image actual binary

    binary_image = 1-binary_image # Invert it, lungs are now 1

 

    # Remove other air pockets inside body

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: # There are air pockets

        binary_image[labels != l_max] = 0

 

    return binary_image
# get masks 

segmented_lungs = segment_lung_mask(patient_pixels, fill_lung_structures=False)

segmented_lungs_fill = segment_lung_mask(patient_pixels, fill_lung_structures=True)

internal_structures = segmented_lungs_fill - segmented_lungs



# isolate lung from chest

copied_pixels = copy.deepcopy(patient_pixels)

for i, mask in enumerate(segmented_lungs_fill): 

    get_high_vals = mask == 0

    copied_pixels[i][get_high_vals] = 0

seg_lung_pixels = copied_pixels

# sanity check

f, ax = plt.subplots(1,2, figsize=(10,6))

ax[0].imshow(patient_pixels[80], cmap=plt.cm.bone)

ax[0].axis(False)

ax[0].set_title('Original')

ax[1].imshow(seg_lung_pixels[80], cmap=plt.cm.bone)

ax[1].axis(False)

ax[1].set_title('Segmented')

plt.show()
f, ax = plt.subplots(2,2, figsize = (10,10))



# pick random slice 

slice_id = 80



ax[0,0].imshow(patient_pixels[slice_id], cmap=plt.cm.bone)

ax[0,0].set_title('Original Dicom')

ax[0,0].axis(False)





ax[0,1].imshow(segmented_lungs_fill[slice_id], cmap=plt.cm.bone)

ax[0,1].set_title('Lung Mask')

ax[0,1].axis(False)



ax[1,0].imshow(seg_lung_pixels[slice_id], cmap=plt.cm.bone)

ax[1,0].set_title('Segmented Lung')

ax[1,0].axis(False)



ax[1,1].imshow(seg_lung_pixels[slice_id], cmap=plt.cm.bone)

ax[1,1].imshow(internal_structures[slice_id], cmap='jet', alpha=0.7)

ax[1,1].set_title('Segmentation with \nInternal Structure')

ax[1,1].axis(False)



# slide through dicom images using a slide bar 

plt.figure(1)

def dicom_animation(x):

    plt.imshow(patient_pixels[x], cmap = plt.cm.gray)

    return x

interact(dicom_animation, x=(0, len(patient_pixels)-1))
import imageio

from IPython import display

print('Original Image Slices before processing')

imageio.mimsave(f'./{patient_id}.gif', patient_pixels, duration=0.1)

display.Image(f'./{patient_id}.gif', format='png')
print('Lung Segmentation Mask')

imageio.mimsave(f'./{patient_id}.gif', segmented_lungs_fill, duration=0.1)

display.Image(f'./{patient_id}.gif', format='png')
print('Segmented Part of Lung Tissue')

imageio.mimsave(f'./{patient_id}.gif', seg_lung_pixels, duration=0.1)

display.Image(f'./{patient_id}.gif', format='png')
from skimage.morphology import opening, closing

from skimage.morphology import disk



def plot_comparison(original, filtered, filter_name):



    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,

                                   sharey=True)

    ax1.imshow(original, cmap=plt.cm.gray)

    ax1.set_title('original')

    ax1.axis('off')

    ax2.imshow(filtered, cmap=plt.cm.gray)

    ax2.set_title(filter_name)

    ax2.axis('off')




original = segmented_lungs_fill[65]



rows = 4

cols = 4

f, ax = plt.subplots(rows, cols, figsize = (15,12))



for i in range(rows*cols):

    if i==0:

        ax[0,0].imshow(original, cmap = plt.cm.gray)

        ax[0,0].set_title('Original')

        ax[0,0].axis(False)

    else:

        closed = closing(original, disk(i))

        ax[int(i/rows),int(i % rows)].set_title(f'closed disk({i})')

        ax[int(i/rows),int(i % rows)].imshow(closed, cmap = plt.cm.gray)

        ax[int(i/rows),int(i % rows)].axis('off')

plt.show()   

    
original_image = patient_pixels[65]

original = segmented_lungs_fill[65]

f, ax = plt.subplots(rows, cols, figsize = (15,15))



for i in range(rows*cols):

    if i==0:

        ax[0,0].imshow(original_image, cmap = plt.cm.gray)

        ax[0,0].set_title('Original')

        ax[0,0].axis(False)

    else:

        closed = closing(original, disk(i))

        ax[int(i/rows),int(i % rows)].set_title(f'closed disk({i})')

        ax[int(i/rows),int(i % rows)].imshow(original_image * closed, cmap = plt.cm.gray)

        ax[int(i/rows),int(i % rows)].axis('off')

plt.show()   