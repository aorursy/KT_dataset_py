import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os

import cv2

import shutil

from matplotlib.figure import Figure

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import scipy.ndimage as ndimage

import scipy

import matplotlib.pyplot as plt

from skimage import measure, morphology, segmentation



# Load the scans in given folder path

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    #'''

    # Convert to Hounsfield units (HU)

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    window_center = -600

    window_width = 1600

    min_value = (2*window_center - window_width)/2.0 - 0.5

    max_value = (2*window_center + window_width)/2.0 - 0.5

    

    dFactor = 255.0/(max_value - min_value)

    #'''

    image[image < min_value] = 0

    image[image > max_value] = 255

    image = (image - min_value)*dFactor

    #image = image.astype(np.uint8)

    image = image.astype(np.uint8)

    

    image[image < 0] = 0

    image[image > 255] = 255

    

    dx = 70

    dy = 43

    dw = 512

    

    image = image[:,dx:dw-dx,dy:dw-dy]

    return np.array(image, dtype=np.uint16)



# Script starts Here.

INPUT_FOLDER = "../input/sample_images"



patients = os.listdir(INPUT_FOLDER)

num = 0

for i in range(len(patients)):

    patient = load_scan(INPUT_FOLDER + '/' + patients[i])

    patient_pixels = get_pixels_hu(patient)

    

    plt.imshow(patient_pixels[80], cmap=plt.cm.gray)

    plt.show()