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

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

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
first_patient = load_scan(INPUT_FOLDER + patients[1])

first_patient_pixels = get_pixels_hu(first_patient)

plt.hist(first_patient_pixels.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



# Show some slice in the middle

plt.imshow(first_patient_pixels[110], cmap=plt.cm.gray)

plt.show()
r1=1   #this is the first image

r2=120   #this is the last image

first_patient_merge=first_patient_pixels[r1]

for i in range (r1,r2):

    first_patient_merge=np.maximum(first_patient_merge,first_patient_pixels[i])



plt.imshow(first_patient_merge, cmap=plt.cm.gray)

plt.show()



plt.hist(first_patient_merge.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



roi=[first_patient_merge <= -100]
r1=80   #this is the first image to merge

r2=90   #this is the last image to merge

first_patient_merge=first_patient_pixels[r1]

for i in range (r1,r2):

    first_patient_merge=np.maximum(first_patient_merge,first_patient_pixels[i])



plt.imshow(first_patient_merge, cmap=plt.cm.gray)

plt.show()



plt.hist(first_patient_merge.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



first_patient_merge2=first_patient_merge

first_patient_merge2[roi]=-2000

first_patient_merge2[first_patient_merge2 <= -1000] = 2000

first_patient_merge2[first_patient_merge2 <= -500] = -2000

first_patient_merge2[first_patient_merge2 > -500] = 2000



plt.imshow(first_patient_merge2, cmap=plt.cm.gray)

plt.show()
first_patient_empty=first_patient_pixels[r1]

for i in range (r1,r2):

    first_patient_empty=np.minimum(first_patient_empty,first_patient_pixels[i])



plt.imshow(first_patient_empty, cmap=plt.cm.gray)

plt.show()



plt.hist(first_patient_empty.flatten(), bins=80, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()



first_patient_empty2=first_patient_empty

first_patient_empty2[roi]=-2000

first_patient_empty2[first_patient_empty2 <= -1000] = 2000

first_patient_empty2[first_patient_empty2 <= -500] = -2000

first_patient_empty2[first_patient_empty2 > -500] = 2000



plt.imshow(first_patient_empty2, cmap=plt.cm.gray)

plt.show()
first_patient_diff3=np.maximum(first_patient_merge2,first_patient_empty2)

plt.imshow(first_patient_diff3, cmap=plt.cm.gray)

plt.show()
def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    

    return image, new_spacing
pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])

print("Shape before resampling\t", first_patient_pixels.shape)

print("Shape after resampling\t", pix_resampled.shape)
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
segmented_lungs = segment_lung_mask(pix_resampled, False)

segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
MIN_BOUND = -1000.0

MAX_BOUND = 400.0

    

def normalize(image):

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)

    image[image>1] = 1.

    image[image<0] = 0.

    return image
PIXEL_MEAN = 0.25



def zero_center(image):

    image = image - PIXEL_MEAN

    return image