%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os

import scipy.ndimage

import matplotlib.pyplot as plt



from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Some constants 

INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER)

patients.sort()



# Load the scans in given folder path

def load_scan(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    

    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    

    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

      

    intercept = scans[0].RescaleIntercept

    image += int(intercept)

    

    return np.array(image)



first_patient = load_scan(INPUT_FOLDER + patients[0])

first_patient_pixels = get_pixels_hu(first_patient)

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



# Show some slice in the middle

plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)

plt.show()