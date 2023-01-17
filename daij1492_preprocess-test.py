import os

os.listdir('../input/sample_images/')
import dicom
# use DeepMan's script here

def load_scan_as_HU_nparray(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

    

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
sample_scan_path = '../input/sample_images/0a0c32c9e08cc2ea76a71649de56be6d'
import numpy as np

sample_scan = load_scan_as_HU_nparray(sample_scan_path)
import matplotlib.pyplot as plt

plt.imshow(sample_scan[49])
# A little dissection of DeepMan's "seperate_lungs_and_pad" function

image = sample_scan[55]
plt.imshow(image)
marker_internal = image < -400
marker_internal
plt.imshow(marker_internal)
from skimage import segmentation

marker_internal = segmentation.clear_border(marker_internal)
plt.imshow(marker_internal)
from skimage import measure

marker_internal_labels = measure.label(marker_internal)
marker_internal_labels
plt.imshow(marker_internal_labels)
regions = measure.regionprops(marker_internal_labels)
dir(regions[0])
regions[0].area
regions[0].coords[:20,:]
areas = [r.area for r in regions]
areas
areas.sort()
areas
areas[-2]
marker_internal_labels.max()
marker_internal_labels.min()
marker_internal_labels_tst = np.copy(marker_internal_labels)

for coordinates in regions[0].coords:

    marker_internal_labels_tst[coordinates[0], coordinates[1]] = 0
plt.hist(marker_internal_labels.flatten())
plt.hist(marker_internal_labels_tst.flatten())
plt.imshow(marker_internal_labels_tst)
marker_internal_labels_tst2 = np.copy(marker_internal_labels)

for region in regions:

    if region.area < areas[-2]:

        for coordinates in region.coords:                

            marker_internal_labels_tst2[coordinates[0], coordinates[1]] = 0
plt.imshow(marker_internal_labels_tst2)
marker_internal = marker_internal_labels_tst2 > 0
plt.imshow(marker_internal)
# Test the whole segmentation function

def seperate_lungs_and_pad(scan):

    

    # make total 256 slices fill in -1100 as exterme value 

    segmented_scan = np.full ((256, 512, 512), THRESHOLD_LOW)

    

    for i, image in enumerate (scan):

        

        # Ignore all slices later than 255 if required.

        if (i == 256):

            break

        

        # Creation of the internal Marker

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

        #Creation of the external Marker

        external_a = ndimage.binary_dilation(marker_internal, iterations=10)

        external_b = ndimage.binary_dilation(marker_internal, iterations=55)

        marker_external = external_b ^ external_a

        #Creation of the Watershed Marker matrix

        marker_watershed = np.zeros((512, 512), dtype=np.int)

        marker_watershed += marker_internal * 255

        marker_watershed += marker_external * 128



        #Creation of the Sobel-Gradient

        sobel_filtered_dx = ndimage.sobel(image, 1)

        sobel_filtered_dy = ndimage.sobel(image, 0)

        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)

        sobel_gradient *= 255.0 / np.max(sobel_gradient)



        #Watershed algorithm

        watershed = morphology.watershed(sobel_gradient, marker_watershed)



        #Reducing the image created by the Watershed algorithm to its outline

        outline = ndimage.morphological_gradient(watershed, size=(3,3))

        outline = outline.astype(bool)



        #Performing Black-Tophat Morphology for reinclusion

        #Creation of the disk-kernel and increasing its size a bit

        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],

                           [0, 1, 1, 1, 1, 1, 0],

                           [1, 1, 1, 1, 1, 1, 1],

                           [1, 1, 1, 1, 1, 1, 1],

                           [1, 1, 1, 1, 1, 1, 1],

                           [0, 1, 1, 1, 1, 1, 0],

                           [0, 0, 1, 1, 1, 0, 0]]

        blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)

        #Perform the Black-Hat

        outline += ndimage.black_tophat(outline, structure=blackhat_struct)



        #Use the internal marker and the Outline that was just created to generate the lungfilter

        lungfilter = np.bitwise_or(marker_internal, outline)

        #Close holes in the lungfilter

        #fill_holes is not used here, since in some slices the heart would be reincluded by accident

        lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)



        #Apply the lungfilter (note the filtered areas being assigned 30 HU)

        segmented_scan[i] = np.where(lungfilter == 1, image, 30*np.ones((512, 512)))

        

    return segmented_scan
THRESHOLD_LOW = -1100

import scipy.ndimage as ndimage

from skimage import morphology

#processed_scan = seperate_lungs_and_pad (sample_scan)



# Kernel dead on this step

# "The kernel was stopped, for exceeding the limits on idle time (20 minutes), storage (512 Mb) or memory (8Gb). You can use the restart button in the toolbar to try again..."
