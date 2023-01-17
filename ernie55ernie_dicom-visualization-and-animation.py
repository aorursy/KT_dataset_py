%matplotlib inline



import matplotlib.pyplot as plt

import dicom

import numpy as np

import os

import scipy.ndimage

from skimage import measure

import matplotlib.animation as ani



# 1 specific patient with 801 ordered CT

# PATIENT_FOLDER = 'D:\image\CTSeries801Ordered'



# 1595 patients with varying number of CT images

PATIENTS_FOLDER = '../input/sample_images/'

patients = os.listdir(PATIENTS_FOLDER)

patients.sort()
def load_scan(path):

    # InvalidDicomError: File is missing 'DICM' marker. Use force=True to force reading

    # MemoryError

    # slices = [dicom.read_file(path + '/' + s, force = True) for s in os.listdir(path)[400:499]]

    slices = [dicom.read_file(path + '/' + s, force = True) for s in os.listdir(path)]

    # Instance number identifies the sequence of the images in a series

    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:

        # Image position of a patient is the x, y, and z coordinates of the upper left hand corner 

        # (center of the first voxel transmitted) in mm

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        # Slice location is the relative position of the image plane expressed in mm

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    

    for s in slices:

        s.SliceThickness = slice_thickness

    return slices
def get_pixels(scans):

    image = np.stack([s.pixel_array for s in scans])

    image = image.astype(np.int16)

    

    image[image == -2000] = 0

    

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(int16)

    return image
# patient = load_scan(PATIENT_FOLDER)

patient = load_scan(PATIENTS_FOLDER + patients[0])

patient_pixels = get_pixels(patient)



# bin width, and color cyan

plt.hist(patient_pixels.flatten(), bins = 80, color = 'c')

plt.xlabel('Hounsfield Units (HU)')

plt.ylabel('Frequency')

plt.show()



# show the slice in the middle

plt.imshow(patient_pixels[60], cmap=plt.cm.gray)

plt.show()
def resample(image, scan, new_spacing = [1, 1, 1]):

    # Currnet pixel spacing

    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))

    spacing = np.array(list(spacing))

    

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    

    return image, new_spacing
image_resampled, spacing = resample(patient_pixels, patient, [1, 1, 1])

print('Shape before resampling\t', patient_pixels.shape)

print('Shape after resampling\t', image_resampled.shape)
def animation(patient_pixels, gif_name):

    fig = plt.figure()

    anim = plt.imshow(patient_pixels[0], cmap=plt.cm.gray)

    plt.grid(False)

    def update(i):

        anim.set_array(patient_pixels[i])

        return anim,

    

    a = ani.FuncAnimation(fig, update, frames=range(len(patient_pixels)), interval=50, blit=True)

    a.save(gif_name, writer='imagemagick')



animation(patient_pixels, 'original_patient.gif')