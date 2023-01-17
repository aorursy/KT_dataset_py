# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import dicom

import os

import scipy.ndimage

import matplotlib.pyplot as plt



from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER)

patients.sort()
type(patients)
len(patients)
patients
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
slices = [dicom.read_file(INPUT_FOLDER + patients[0] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[0])]
type(slices)
len(slices)
type(slices[0])
slices[0]
slices = [dicom.read_file(INPUT_FOLDER + patients[1] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[1])]

len(slices)
slices = [dicom.read_file(INPUT_FOLDER + patients[2] + '/' + s) for s in os.listdir(INPUT_FOLDER + patients[2])]

len(slices)
slices[0]
slices[0].SOPClassUID
slices[1].SliceLocation
slices[0].ImagePositionPatient[2]
slices[0].PatientName
type(slices[1].PixelData)
def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

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
first_patient = load_scan(INPUT_FOLDER + patients[0])

first_patient_pixels = get_pixels_hu(first_patient)

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



# Show some slice in the middle

plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)

plt.show()
for i in range(len(slices)):

    print(i,'\t',slices[i].ImagePositionPatient)
slices[1].ImagePositionPatient
slices[1].pixel_array