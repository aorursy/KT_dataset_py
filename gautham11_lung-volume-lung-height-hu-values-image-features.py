%matplotlib inline



import os

import cv2



import pydicom

import pandas as pd

import numpy as np 

import torch

import scipy

import tensorflow as tf 

import matplotlib.pyplot as plt 

from pathlib import Path

import scipy.ndimage as ndimage

from skimage import measure, morphology, segmentation

from scipy.ndimage.interpolation import zoom

from PIL import Image 

import time



from tqdm.notebook import tqdm

%matplotlib inline



from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
data_dir = Path('../input/osic-pulmonary-fibrosis-progression')

patient_paths = list((data_dir/'train').glob('*'))

sample_patient = Path('../input/osic-pulmonary-fibrosis-progression/train/ID00233637202260580149633')


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

    marker_watershed = np.zeros(image.shape, dtype=np.int)

    marker_watershed += marker_internal * 255

    marker_watershed += marker_external * 128

    

    return marker_internal, marker_external, marker_watershed





def seperate_lungs(image,iterations = 1):

    """

    Segments lungs using various techniques.

    

    Parameters: image (Scan image), iterations (more iterations, more accurate mask)

    

    Returns: 

        - Segmented Lung

        - Lung Filter

        - Outline Lung

        - Watershed Lung

        - Sobel Gradient

    """

    

    # Store the start time

    # start = time.time()

    marker_internal, marker_external, marker_watershed = generate_markers(image)

    

    

    '''

    Creation of Sobel Gradient

    '''

    

    # Sobel-Gradient

    sobel_filtered_dx = ndimage.sobel(image, 1)

    sobel_filtered_dy = ndimage.sobel(image, 0)

    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)

    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    

    

    '''

    Using the watershed algorithm

    

    

    We pass the image convoluted by sobel operator and the watershed marker

    to morphology.watershed and get a matrix matrix labeled using the 

    watershed segmentation algorithm.

    '''

    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    

    '''

    Reducing the image to outlines after Watershed algorithm

    '''

    outline = ndimage.morphological_gradient(watershed, size=(3,3))

    outline = outline.astype(bool)

    

    

    '''

    Black Top-hat Morphology:

    

    The black top hat of an image is defined as its morphological closing

    minus the original image. This operation returns the dark spots of the

    image that are smaller than the structuring element. Note that dark 

    spots in the original image are bright spots after the black top hat.

    '''

    

    # Structuring element used for the filter

    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],

                       [0, 1, 1, 1, 1, 1, 0],

                       [1, 1, 1, 1, 1, 1, 1],

                       [1, 1, 1, 1, 1, 1, 1],

                       [1, 1, 1, 1, 1, 1, 1],

                       [0, 1, 1, 1, 1, 1, 0],

                       [0, 0, 1, 1, 1, 0, 0]]

    

    blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)

    

    # Perform Black Top-hat filter

    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    

    '''

    Generate lung filter using internal marker and outline.

    '''

    lungfilter = np.bitwise_or(marker_internal, outline)

    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

    

    '''

    Segment lung using lungfilter and the image.

    '''

    segmented = np.where(lungfilter, image, -1000)

    

    #return segmented, lungfilter, outline, watershed, sobel_gradient

    return segmented
def fix_pxrepr(dcm):

#     if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100:

#         return dcm

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    #dcm.RescaleIntercept = -1000

    return dcm



def load_scan(path):

    paths = os.listdir(path)

    paths = sorted(paths, key=lambda x: int(str(x).split('/')[-1].split('.')[0]))

    slices = [pydicom.read_file(path + '/' + s) for s in paths]

    #slices = list(map(fix_pxrepr, slices))

    try:

        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    except:

        pass

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        try:

            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        except:

            slice_thickness = slices[0].SliceThickness

    for s in slices:

        s.SliceThickness = slice_thickness

    return slices
def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    image[image <= -1900] = -1000

    return np.array(image, dtype=np.int16)
def resample(image, scan, new_spacing=[1,1,1]):

    st = time.time()

    slice_thickness = scan[0].SliceThickness

    spacing = np.array([slice_thickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    print('resample factor', time.time() - st, real_resize_factor)

    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')

    print('resample time', time.time() - st)

    return image, new_spacing



def torch_resample(image, scan, new_spacing=[1,1,1]):

    st = time.time()

    slice_thickness = scan[0].SliceThickness

    spacing = np.array([slice_thickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

    image = torch.nn.functional.interpolate(image, scale_factor=tuple(real_resize_factor), mode='nearest')

    image = image.squeeze(0).squeeze(0).numpy()

    return image, new_spacing









def get_3d_resampled_array(patient_path):

    start_time = time.time()

    patient_slices = load_scan(str(patient_path))

    patient_slices_hu = get_pixels_hu(patient_slices)

    print('HU loaded', time.time() - start_time)

    print(patient_slices_hu.shape)

    #lungmask_3d = np.apply_over_axes(seperate_lungs, patient_slices_hu, 0)

    idx = np.ndindex(patient_slices_hu.shape[0])

    patient_slices_hu_masked = np.zeros(patient_slices_hu.shape)

    for i in idx:

        patient_slices_hu_masked[i] = seperate_lungs(patient_slices_hu[i])

        #patient_slices_hu_masked[i, :, :] = np.where(lungmask, patient_slices_hu[i, :, :], -1000)

    #patient_slices_hu_masked = np.where(lungmask_3d, patient_slices_hu, -1000)

    

    print('mask generated', time.time() - start_time)

    resampled_array, spacing = torch_resample(patient_slices_hu_masked, patient_slices, [1,1,1])

    print('after resample', time.time() - start_time)

    return resampled_array, spacing





def get_features_from_3d_array(resampled_array, spacing):

    features = {}

    

    # volume of lungs

    cube_volume = spacing[0] * spacing[1] * spacing[2]

    total_lung_volume = (resampled_array[resampled_array > -900].shape[0] * cube_volume)

    lung_volume_in_liters = total_lung_volume / (1000*1000)

    features['lung_volume_in_liters'] = lung_volume_in_liters

    

    #HU unit binning

    bins_threshold = (resampled_array <= 300) & (resampled_array >= -900)

    total_hu_units_bin = resampled_array[bins_threshold].flatten().shape[0]

    bin_values, bins = np.histogram(resampled_array[bins_threshold].flatten(), bins=range(-900, 400, 100))

    features['total_hu_units_bin'] = total_hu_units_bin

    for i, _bin in enumerate(bins[:-1]):

        features[f'bin_{_bin}'] = bin_values[i] / total_hu_units_bin

    

    #mean, skew, kurtosis

    lung_threshold = (resampled_array <= -320) & (resampled_array >= -900)

    histogram_values, _ = np.histogram(resampled_array[lung_threshold].flatten(), bins=100)

    features['lung_mean_hu'] = np.mean(resampled_array[lung_threshold].flatten())

    features['lung_skew'] = skew(histogram_values)

    features['lung_kurtosis'] = kurtosis(histogram_values)

    

    #height_of_lung

    n_lung_pixels = lung_threshold.sum(axis=1).sum(axis=1)

    height_start = np.argwhere(n_lung_pixels > 1000).min()

    height_end = np.argwhere(n_lung_pixels > 1000).max()

    features['height_of_lung_cm'] = (height_end - height_start)/10

    

    return features
start_time = time.time()

first_patient = load_scan(str(sample_patient))

first_patient_pixels = get_pixels_hu(first_patient)

print('number of slices', len(first_patient))



first_patient_masked_pixels = np.zeros(first_patient_pixels.shape)

lungmasks = np.zeros(first_patient_pixels.shape)



for i in range(first_patient_pixels.shape[0]):

    semented_lung = seperate_lungs(first_patient_pixels[i, :, :])

    first_patient_masked_pixels[i, :, :] = semented_lung



    

print('time taken', time.time() - start_time)

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()
slice_n = 220

# Show some slice in the middle

plt.imshow(first_patient_pixels[slice_n], cmap=plt.cm.gray)

plt.show()



plt.imshow(first_patient_masked_pixels[slice_n], cmap=plt.cm.gray)

plt.show()
_ = plt.hist(first_patient_masked_pixels[slice_n].flatten(), bins=80)
slice_n = 220

threshold_filter = (first_patient_masked_pixels[slice_n] >= -900)# & (first_patient_masked_pixels[slice_n] <= -320)

plt.imshow(np.where(threshold_filter, first_patient_masked_pixels[slice_n], -1000), cmap=plt.cm.gray)

plt.show()



plt.imshow(first_patient_masked_pixels[slice_n], cmap=plt.cm.gray)

plt.show()



plt.imshow(first_patient_pixels[slice_n], cmap=plt.cm.gray)

plt.show()
start_time = time.time()

pix_resampled, spacing = resample(first_patient_masked_pixels, first_patient, [1,1,1])

print('time taken', time.time() - start_time)

print("Shape before resampling\t", first_patient_pixels.shape)

print("Shape after resampling\t", pix_resampled.shape)
start_time = time.time()

pix_resampled, spacing = torch_resample(first_patient_masked_pixels, first_patient, [1,1,1])

print('time taken', time.time() - start_time)

print("Shape before resampling\t", first_patient_pixels.shape)

print("Shape after resampling\t", pix_resampled.shape)
spacing
resampled_array, spacing = get_3d_resampled_array(str(patient_paths[6]))

print(resampled_array.shape)

print(spacing)
cube_volume = spacing[0] * spacing[1] * spacing[2]
total_lung_volume = (resampled_array[resampled_array > -900].shape[0] * cube_volume)

lung_volume_in_liters = total_lung_volume / (1000*1000)

print(total_lung_volume, lung_volume_in_liters)
bins_threshold = (resampled_array <= 300) & (resampled_array >= -900)

bin_values, bins = np.histogram(resampled_array[bins_threshold].flatten(), bins=range(-900, 400, 100))

print(bin_values)

print(bins)
list(bins)
bin_values / sum(bin_values)
lung_threshold = (resampled_array <= -320) & (resampled_array >= -900)

histogram_values, _ = np.histogram(resampled_array[lung_threshold].flatten(), bins=100)
_ = plt.hist(resampled_array[lung_threshold].flatten(), bins=100)
np.mean(resampled_array[lung_threshold].flatten())
from scipy.stats import kurtosis, skew



print(skew(histogram_values))

print(kurtosis(histogram_values))
plt.plot(lung_threshold.sum(axis=1).sum(axis=1))
height_start = np.argwhere(lung_threshold.sum(axis=1).sum(axis=1) > 1000).min()

height_end = np.argwhere(lung_threshold.sum(axis=1).sum(axis=1) > 1000).max()

print(height_start, height_end)



height = height_end - height_start



print(height)
from scipy.stats import kurtosis, skew



def get_features_from_3d_array(resampled_array, spacing):

    features = {}

    

    # volume of lungs

    cube_volume = spacing[0] * spacing[1] * spacing[2]

    total_lung_volume = (resampled_array[resampled_array > -900].shape[0] * cube_volume)

    lung_volume_in_liters = total_lung_volume / (1000*1000)

    features['lung_volume_in_liters'] = lung_volume_in_liters

    

    #HU unit binning

    bins_threshold = (resampled_array <= 300) & (resampled_array >= -900)

    total_hu_units_bin = resampled_array[bins_threshold].flatten().shape[0]

    bin_values, bins = np.histogram(resampled_array[bins_threshold].flatten(), bins=range(-900, 400, 100))

    features['total_hu_units_bin'] = total_hu_units_bin

    for i, _bin in enumerate(bins[:-1]):

        features[f'bin_{_bin}'] = bin_values[i] / total_hu_units_bin

    

    #mean, skew, kurtosis

    lung_threshold = (resampled_array <= -320) & (resampled_array >= -900)

    histogram_values, _ = np.histogram(resampled_array[lung_threshold].flatten(), bins=100)

    features['lung_mean_hu'] = np.mean(resampled_array[lung_threshold].flatten())

    features['lung_skew'] = skew(histogram_values)

    features['lung_kurtosis'] = kurtosis(histogram_values)

    

    #height_of_lung

    n_lung_pixels = lung_threshold.sum(axis=1).sum(axis=1)

    height_start = np.argwhere(n_lung_pixels > 1000).min()

    height_end = np.argwhere(n_lung_pixels > 1000).max()

    features['height_of_lung_cm'] = (height_end - height_start)/10

    

    return features
pd.DataFrame(pd.Series(get_features_from_3d_array(resampled_array, spacing)))
import traceback

import warnings



warnings.filterwarnings('ignore')



patients_feature_df = pd.DataFrame()

st = time.time()



# Remove filter to run for all patients

for patient_path in patient_paths[:5]:

    try:

        resampled_array, spacing = get_3d_resampled_array(str(patient_path))

        features = get_features_from_3d_array(resampled_array, spacing)

        features['patient_id'] = str(patient_path).split('/')[-1]

        features['missing'] = 0

    except Exception as e:

        features = {}

        features['missing'] = 1

        print(e)

    patient_df = pd.DataFrame(pd.Series(features)).T

    patients_feature_df = pd.concat([patients_feature_df, patient_df], ignore_index=True)

    patients_feature_df.to_csv('patient_feature2_df.csv', index=False)

        

print('Total Time', time.time() - st)
patients_feature_df