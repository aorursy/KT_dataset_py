# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dicom

import os

import time

import scipy.ndimage

import matplotlib.pyplot as plt

from skimage.segmentation import clear_border

from skimage.morphology import disk, dilation, binary_erosion, remove_small_objects, binary_closing

from skimage.measure import label, regionprops

#from skimage.morphology import binary_dilation, binary_opening

from skimage.filters import roberts

from scipy import ndimage as ndi
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

    

    # not actually binary, but 1 and 2. 

    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -320, dtype=np.int8)+1

    labels = label(binary_image)

    

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

            labeling = label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

            

            if l_max is not None: #This slice contains some lung

                binary_image[i][labeling != l_max] = 1



    

    binary_image -= 1 #Make the image actual binary

    binary_image = 1-binary_image # Invert it, lungs are now 1

    

    # Remove other air pockets insided body

    labels = label(binary_image, background=0)

    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None: # There are air pockets

        binary_image[labels != l_max] = 0

 

    return binary_image



t0 = time.time()

# Input data files are available in the "../input/" directory.

INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER)

patients.sort()



f, plots = plt.subplots(5, 4, figsize=(20, 20))

nPlots = 20

for i in range(nPlots):

    test_patient_scans = load_scan(INPUT_FOLDER + patients[i])

    test_patient_scans = get_pixels_hu(test_patient_scans)

    ct = segment_lung_mask(test_patient_scans, False)

    dimsum1 = np.sum(ct, 1)

    plots[int(i/4), int(i % 4)].axis('off')

    plots[int(i/4), int(i % 4)].set_title(patients[i])

    plots[int(i/4), int(i % 4)].imshow(dimsum1, aspect='auto', cmap=plt.cm.bone)

    #print("Otro TAC")



t1 = time.time()

print ("Takes {0:.1f} minutes".format((t1-t0)/60)) # 0.7 minutes with simple binary; 8.9 complex
def read_ct_scan(folder_name):

    # Read the slices from the dicom file

    slices = [dicom.read_file(folder_name + filename) for filename in os.listdir(folder_name)]



    # Sort the dicom slices in their respective order

    slices.sort(key=lambda x: int(x.InstanceNumber))



    # Get the pixel values for all the slices

    slices = np.stack([s.pixel_array for s in slices])

    slices[slices == -2000] = 0

    return slices



def get_segmented_lungs0(im):

    '''

    This function segments the lungs from the given 2D slice.

    '''

    binary = im < 604 # Convert into a binary image.

    cleared = clear_border(binary) # Remove the blobs connected to the border of the image



    return cleared



def segment_lung_from_ct_scan0(ct_scan):

    return np.asarray([get_segmented_lungs0(slice) for slice in ct_scan])



def get_segmented_lungs(im):

    '''

    This function segments the lungs from the given 2D slice.

    '''

    binary = im < 604 # Convert into a binary image.

    cleared = clear_border(binary) # Remove the blobs connected to the border of the image

    

    label_image = label(cleared)

    

    areas = [r.area for r in regionprops(label_image)]

    areas.sort()

    if len(areas) > 2:

        for region in regionprops(label_image):

            if region.area < areas[-2]:

                for coordinates in region.coords:                

                       label_image[coordinates[0], coordinates[1]] = 0

    binary = label_image > 0    



    selem = disk(2)

    binary = binary_erosion(binary, selem)



    selem = disk(10)

    binary = binary_closing(binary, selem)

    

    edges = roberts(binary)

    binary = ndi.binary_fill_holes(edges)

    

    get_high_vals = binary == 0

    im[get_high_vals] = 0

    

    return im



def segment_lung_from_ct_scan(ct_scan):

    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

t0 = time.time()

# Input data files are available in the "../input/" directory.

INPUT_FOLDER = '../input/sample_images/'

patients = os.listdir(INPUT_FOLDER)

patients.sort()



f, plots = plt.subplots(5, 4, figsize=(20, 20))

nPlots = 20

for i in range(nPlots):

    test_patient_scans = read_ct_scan(INPUT_FOLDER + patients[i] + "/")

    ct = segment_lung_from_ct_scan0(test_patient_scans)

    dimsum1 = np.sum(ct, 1)

    plots[int(i/4), int(i % 4)].axis('off')

    plots[int(i/4), int(i % 4)].set_title(patients[i])

    plots[int(i/4), int(i % 4)].imshow(dimsum1, aspect='auto', cmap=plt.cm.bone)



t1 = time.time()

print ("Takes {0:.1f} minutes".format((t1-t0)/60)) # 0.7 minutes with simple binary; 8.9 complex