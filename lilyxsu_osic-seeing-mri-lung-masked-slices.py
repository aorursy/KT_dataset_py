# common packages 

import numpy as np 

import os

import copy

from math import *

import matplotlib.pyplot as plt

from functools import reduce

# reading in dicom files

import pydicom

# skimage image processing packages

from skimage import measure, morphology

from skimage.morphology import ball, binary_closing

from skimage.measure import label, regionprops

# scipy linear algebra functions 

from scipy.linalg import norm

import scipy.ndimage

# plotly 3D interactive graphs 

import plotly

from plotly.graph_objs import *

import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def get_dicom_files(folder):

    files_in_folder = os.listdir(folder)

    files_in_folder = [folder + i for i in files_in_folder]

    return files_in_folder



train = '../input/osic-pulmonary-fibrosis-progression/train/'

file_folders = get_dicom_files(train)

# file_folders[:3]



def load_scan(path): 

    '''Load all DICOM images from a folder into a list for manipulation.

    Infer slice thickness '''

    slices = [pydicom.read_file(path+'/'+s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def get_pixels_hu(scans):

    '''Pixels outside cylindrical scanning bounds are set to Hounsfield 

    Units of air. 

    Convert to Hounsfield units by multiplying a rescale slope and adding

    an intercept from teh metadata of scans.

    '''

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



def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]

    vals = vals[vals != bg]

    if len(counts) > 0:

        return vals[np.argmax(counts)]

    else:

        return None

def segment_lung_mask(image, fill_lung_structures=True):

    ''' Isolate just the lungs and some tissue around it by masking

    out everything else'''

    # not actually binary, but 1 and 2. 

    # 0 is treated as background, which we do not want

    binary_image = np.array(image >= -700, dtype=np.int8)+1

    labels = measure.label(binary_image)

 

    # Pick the pixel in the very corner to determine which label is     air.

    # Improvement: Pick multiple background labels from around the    patient

    # More resistant to “trays” on which the patient lays cutting   the air around the person in half

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





# set path and load files 

path = file_folders[3]

patient_dicom = load_scan(path)

patient_pixels = get_pixels_hu(patient_dicom)





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

plt.imshow(seg_lung_pixels[326], cmap=plt.cm.bone)