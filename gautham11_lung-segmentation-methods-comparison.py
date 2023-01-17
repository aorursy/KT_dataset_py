%matplotlib inline



import numpy as np

import pandas as pd

import pydicom

import os

import scipy.ndimage as ndimage

from skimage import measure, morphology, segmentation

import matplotlib.pyplot as plt

import os

from pathlib import Path

import cv2



import time
DATA_DIR = Path('/kaggle/input/osic-pulmonary-fibrosis-progression/')
!pip install ../input/lungmask/lungmask/dependencies/SimpleITK-1.2.4-cp37-cp37m-manylinux1_x86_64.whl

!pip install ../input/lungmask/lungmask/dependencies/fastremap-1.10.2-cp37-cp37m-manylinux1_x86_64.whl

!pip install ../input/lungmask/lungmask/dependencies/fill_voids-2.0.0-cp37-cp37m-manylinux1_x86_64.whl
!pip install ../input/lungmask/lungmask
# Load the scans in given folder path

def load_scan(path):

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:

        s.SliceThickness = slice_thickness

    return slices



#to HU values

def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)



# the scans can have different pixel size to real world mapping

# resample to fix it to 1

def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = np.array([float(scan[0].SliceThickness)] + list(scan[0].PixelSpacing), dtype=np.float32)



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    

    return image, new_spacing
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

    binary_image = np.array(image > -320, dtype=np.int8)+1

    labels = measure.label(binary_image)

    

    # Pick the pixel in the very corner to determine which label is air.

    #   Improvement: Pick multiple background labels from around the patient

    #   More resistant to "trays" on which the patient lays cutting the air 

    #   around the person in half

    background_label = labels[0,0,0]

    

    #Fill the air around the person

    binary_image[background_label == labels] = 2

    

    

    # Method of filling the lung structures (that is superior to something like 

    # morphological closing)

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

    

    # Remove other air pockets insided body

    labels = measure.label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: # There are air pockets

        binary_image[labels != l_max] = 0

 

    return binary_image



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
def seperate_lungs(image, iterations = 1):

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

    start = time.time()

    

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

    segmented = np.where(lungfilter == 1, image, -2000*np.ones(image.shape))

    

    return segmented, lungfilter, outline, watershed, sobel_gradient

from lungmask import mask

import SimpleITK as sitk
mask_model = mask.load_model('unet','R231', '../input/lungmask/lungmask/models/unet_r231-d5d2fc3d.pth')
def get_img_sitk(path):

    return sitk.ReadImage(path)



def generate_mask_image(path):

    img = get_img_sitk(path)

    segmentation = (mask.apply(img, mask_model)[0,:,:])

    segmentation[segmentation > 0] = 1

    segmentation = cv2.resize(segmentation, (512, 512))

    img_array = cv2.resize(((sitk.GetArrayFromImage(img)[0,:,:]) - float(img.GetMetaData('0028|1052'))) / (float(img.GetMetaData('0028|1053')) * 1000), (512, 512))

    masked_img = np.where(segmentation == 1, img_array, 0)

    return segmentation, masked_img
MIN_BOUND = -1000.0

MAX_BOUND = 400.0

    

def normalize(image):

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    image[image>1] = 1.

    image[image<0] = 0.

    return image



##Zero centering

PIXEL_MEAN = 0.25



def zero_center(image):

    image = image - PIXEL_MEAN

    return image
patient_id = 'ID00007637202177411956430'
train = pd.read_csv(DATA_DIR/'train.csv')

train[train['Patient'] == patient_id]
slices = load_scan(str(DATA_DIR/f'train/{patient_id}'))

hu_slices = get_pixels_hu(slices)
def plot_hist(arr):

    plt.hist(arr.flatten(), bins=80, color='c')

    plt.xlabel("Hounsfield Units (HU)")

    plt.ylabel("Frequency")

    plt.show()



# Show some slice in the middle

def plot_slice(arr, n=80):

    plt.imshow(arr[n], cmap=plt.cm.gray)

    plt.show()

    

#watershed plot markers

def plot_watershed_markers(hu_slice):

    test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(hu_slice)



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

    

def plot_watershed_results(hu_slice, itrs=1):

    test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient = seperate_lungs(hu_slice, itrs)

    f, ax = plt.subplots(3, 2, sharey=True, figsize = (12, 12))

    ax[0][0].imshow(test_sobel_gradient, cmap='gray')

    ax[0][0].set_title("Sobel Gradient")

    ax[0][0].axis('off')



    ax[0][1].imshow(test_watershed, cmap='gray')

    ax[0][1].set_title("Watershed")

    ax[0][1].axis('off')

    

    ax[1][0].imshow(test_segmented, cmap='gray')

    ax[1][0].set_title('Segmented Lung')

    ax[1][0].axis('off')

    

    ax[1][1].imshow(test_lungfilter, cmap='gray')

    ax[1][1].set_title('Lungfilter')

    ax[1][1].axis('off')

    

    ax[2][0].imshow(test_outline, cmap='gray')

    ax[2][0].set_title('Outline')

    ax[2][0].axis('off')



    plt.show()
plot_hist(hu_slices)
plot_slice(hu_slices, 20)
# the images rearraged by instance numbers

plt.imshow(pydicom.read_file(str(DATA_DIR/f'train/{patient_id}/20.dcm')).pixel_array, cmap='gray')
st = time.time()

resampled_slices, new_spacing = resample(hu_slices, slices)

print(new_spacing)

print(resampled_slices.shape)

print(time.time() - st)
plot_slice(resampled_slices,200)
st = time.time()

test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient = seperate_lungs(resampled_slices[200], 1)

print(time.time() - st)
plot_watershed_results(resampled_slices[200])
plot_hist(test_segmented[test_segmented > -1000])
st = time.time()

mask_filter, masked_lung = generate_mask_image(str(DATA_DIR/f'train/{patient_id}/20.dcm'))

print(time.time() - st)
plt.imshow(mask_filter,cmap='gray')
plt.imshow(masked_lung, cmap='gray')
# this is fastest, but doesn't seem reliable, or maybe I'm missing something

# There is no proper segmentation in the source kernel too

# https://www.kaggle.com/allunia/pulmonary-fibrosis-dicom-preprocessing

st = time.time()

segmented_lungs = segment_lung_mask(resampled_slices, False)

print((time.time() - st) / resampled_slices.shape[0])
plt.imshow(segmented_lungs[200, :, :], cmap='gray')