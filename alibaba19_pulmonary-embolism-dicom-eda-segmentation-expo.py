!conda install -c conda-forge gdcm -y
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from pathlib import Path

import matplotlib.pyplot as plt

import pydicom

import cv2

import seaborn as sns

import gdcm

from skimage import measure, segmentation, morphology

from skimage.morphology import disk, opening, closing

from scipy import ndimage
input_path = Path('../input/rsna-str-pulmonary-embolism-detection')

os.listdir(input_path)
train_df = pd.read_csv(input_path/'train.csv')

test_df = pd.read_csv(input_path/'test.csv')

sub_df = pd.read_csv(input_path/'sample_submission.csv')
train_df.shape, test_df.shape, sub_df.shape
train_df.head()
test_df.head()
sub_df.head()
list(train_df.columns)
#let's just compute the % of scans with a particular attribute

positive = train_df['pe_present_on_image'].value_counts()[1] / len(train_df['pe_present_on_image'])

print("{0:.2f}% of the training data shows Pulmonary Embolism visually".format(positive * 100))



motion_issue = (train_df['qa_motion'].value_counts()[1] / len(train_df))

print("{0:.2f}% of the scans are noted that motion may have caused issues".format(motion_issue*100))



contrast_issue = (train_df['qa_contrast'].value_counts()[1]) / len(train_df)

print("{0:.2f}% of the scans are noted with contrast issues".format(contrast_issue*100))





left_pe = (train_df['leftsided_pe'].value_counts())

left_pe_pct = left_pe[1] / len(train_df['leftsided_pe'])

print("{0:.2f}% of the scans are noted with PE on left side".format(left_pe_pct*100))





right_pe = (train_df['rightsided_pe'].value_counts())

right_pe_pct = right_pe[1] / len(train_df['rightsided_pe'])

print("{0:.2f}% of the scans are noted with PE on right side".format(right_pe_pct*100))



central_pe = (train_df['central_pe'].value_counts())

central_pe_pct = central_pe[1] / len(train_df['central_pe'])

print("{0:.2f}% of the scans are noted with PE on right side".format(central_pe_pct*100))



chronic_pe = train_df['chronic_pe'].value_counts()

chronic_pe_pct = chronic_pe[1] / len(train_df['chronic_pe'])

print("{0:.2f}% of the scans feature Chronic PE".format(chronic_pe_pct * 100))



acute_and_chronic = train_df['acute_and_chronic_pe'].value_counts()

acute_chr_pct = acute_and_chronic[1] / len(train_df['acute_and_chronic_pe'])

print("{0:.2f}% of PE present are both acute AND Chronic :(".format(chronic_pe_pct * 100))



indeterminate = train_df['indeterminate'].value_counts()

indeterminate_pct = indeterminate[1] / len(train_df['indeterminate'])

print("{0:.2f}% of scans had QA issues".format(chronic_pe_pct * 100))
#no information on this in competition description

train_df['flow_artifact'].value_counts() 
training_path = input_path/'train'

len(os.listdir(training_path))
%%time

scans_per_folder = []

for x in os.listdir(training_path):

    path = Path(str(training_path) + '/' + str(x))

    scans_per_folder.append(len(os.listdir(path)))
len(scans_per_folder), pd.Series(scans_per_folder).unique()
%%time

slices_per_scan = []

for x in os.listdir(training_path):

    path = Path(str(training_path) + '/' + str(x))

    for folder in os.listdir(path):

        scan_path = Path(str(path)+ '/' + str(folder))

        slices_per_scan.append(len(os.listdir(scan_path)))
plt.title('Distribution of Slices per Scan')

plt.xlabel('Number of slices')

plt.ylabel('Frequency')

plt.hist(slices_per_scan, bins=50);
#returns a list with the dicoms in order

def dcm_sort(scan_path, scan_folder):

    #a list comprehension create distill our exact file paths -- ugh

    dcm_paths = [(str(scan_path) + '/' + file) for file in scan_folder]

    #list comprehension that runs through each slice in the folder

    dcm_stacked = [pydicom.dcmread(dcm) for dcm in dcm_paths]

    dcm_stacked.sort(key=lambda x: int(x.InstanceNumber), reverse=True)

    #returning a python list of dicoms sorted

    return dcm_stacked
scan_path = training_path/'858a11d72ad0/7829612362e8'

scan_folder = os.listdir(scan_path)

print("There are {} slices in the selected scan".format(len(scan_folder)))
%%time

sorted_scan = dcm_sort(scan_path, scan_folder)
sorted_scan[0]
sorted_scan[0].PixelData[0:100]
sorted_scan[0].pixel_array[0:5]
plt.imshow(sorted_scan[100].pixel_array);
#let's concentrate on the a section of 60 slices in this scan

middle_scan = sorted_scan[80:140]



fig,ax = plt.subplots(4,5, figsize=(12,8))

for n in range(4):

    for m in range(5):

        ax[n,m].imshow(middle_scan[n*5+m].pixel_array, cmap='Blues_r')
#let's take a look at a single slice's pixel distribution

plt.hist(middle_scan[20].pixel_array);
one_slice = middle_scan[20].pixel_array

one_slice[one_slice <= -1000] = 0

plt.imshow(one_slice, cmap='Blues_r');
one_slice = middle_scan[20]

one_slice.RescaleIntercept, one_slice.RescaleSlope
def scan_transformed_hu(dcm_sorted, threshold=-1000, replace=-1000):

    intercept = dcm_sorted[0].RescaleIntercept

    slices_stacked = np.stack([dcm.pixel_array for dcm in dcm_sorted])

    slices_stacked = slices_stacked.astype(float)

    

    #converts the unknown values to desired replacement

    slices_stacked[slices_stacked <= threshold] = replace

    

    #turn into hounsfield scale

    slices_stacked += np.int16(intercept)

    

    return np.array(slices_stacked, dtype=np.int16)
middle_slices_hu = scan_transformed_hu(middle_scan, replace=0)



fig,ax = plt.subplots(12,5, figsize=(20,20))

for n in range(12):

    for m in range(5):

        ax[n,m].imshow(middle_slices_hu[n*5+m], cmap='Blues_r')
test_slice = middle_slices_hu[10]

fig, ax = plt.subplots(1,2, figsize=(12,3))

ax[0].imshow(test_slice)

ax[0].set_title('Slice #90') #middle scan was 80-120 and this is 10th one....

ax[1].set_title('Pixel Distribution of Slice')

ax[1].hist(test_slice);
internal_marker = test_slice < -300

internal_marker[203:207, 203:220] #just to show the discrepeny in middle somewhere
#this represents the region we know definitely features lung tissue

plt.title('prelimary internal marker')

plt.imshow(segmentation.clear_border(internal_marker), cmap='gray');
internal_marker_labels = measure.label(segmentation.clear_border(internal_marker))

plt.imshow(internal_marker_labels, cmap='gray');
#explicating the next list comprehension

measure.regionprops(internal_marker_labels)[0:3]
areas = [x.area for x in measure.regionprops(internal_marker_labels)]

areas.sort()

areas
for region in measure.regionprops(internal_marker_labels):

    if region.area < areas[-2]:

        for coordinates in region.coords:

            internal_marker_labels[coordinates[0], coordinates[1]] = 0
marker_internal = internal_marker_labels > 0

plt.title('Internal marker')

plt.imshow(marker_internal, cmap='gray');
external_a = ndimage.binary_dilation(marker_internal, iterations=10)

external_b = ndimage.binary_dilation(marker_internal, iterations=50)

marker_external = external_b ^ external_a

#since they're set to binary values - finding sum will tell you how much white

external_a.sum(), external_b.sum() 
fig, ax = plt.subplots(1, 3, figsize=(20,8))

ax[0].imshow(external_a, cmap='gray')

ax[0].set_title('dilation of internal marker - 10 iterations')

ax[1].imshow(external_b, cmap='gray')

ax[1].set_title('dilation of internal marker - 50 iterations')

ax[2].set_title('external marker')

ax[2].imshow(marker_external, cmap='gray');
watershed_marker = np.zeros((512, 512), dtype=np.int)

watershed_marker += marker_internal * 255 #high intensity

watershed_marker += marker_external * 128 #medium intensity
plt.title('Watershed marker!')

plt.imshow(watershed_marker, cmap='gray');
fig, ax = plt.subplots(1, 2, figsize=(20,8))

ax[0].imshow(ndimage.sobel(test_slice, 0), cmap='gray')

ax[0].set_title('vertical edges')

ax[1].imshow(ndimage.sobel(test_slice, 1), cmap='gray')

ax[1].set_title('horizontal edges');
x_edges = ndimage.sobel(test_slice, 1)

y_edges = ndimage.sobel(test_slice, 0)

sobel_grad = np.hypot(x_edges, y_edges)

sobel_grad *= 255.0 / np.max(sobel_grad)

plt.title('sobel gradient')

plt.imshow(sobel_grad, cmap='gray');
img_watershed = segmentation.watershed(test_slice, watershed_marker)

watershed = segmentation.watershed(sobel_grad, watershed_marker)
fig, ax = plt.subplots(1,2, figsize=(20,8))

ax[0].imshow(img_watershed, cmap='gray')

ax[0].set_title('watershed seg w/ original img')

ax[1].set_title('watershed seg w/ sobel gradient')

ax[1].imshow(watershed, cmap='gray');
#let's try out different kernel sizes :)

fig, ax = plt.subplots(1, 3, figsize=(20,8))

ax[0].imshow(ndimage.morphological_gradient(watershed, size=(2,2)))

ax[0].set_title('outline derived from 2x2 kernel')

ax[1].imshow(ndimage.morphological_gradient(watershed, size=(3,3)))

ax[1].set_title('outline derived from 3x3 kernel')

ax[2].set_title('outline derived from 7x7 kernel')

ax[2].imshow(ndimage.morphological_gradient(watershed, size=(7, 7)));
outline = ndimage.morphological_gradient(watershed, size=(3,3))
#openCV has kernel fxns - not so with scipy, hmm

blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],

                       [0, 1, 1, 1, 1, 1, 0],

                       [1, 1, 1, 1, 1, 1, 1],

                       [1, 1, 1, 1, 1, 1, 1],

                       [1, 1, 1, 1, 1, 1, 1],

                       [0, 1, 1, 1, 1, 1, 0],

                       [0, 0, 1, 1, 1, 0, 0]]



blackhat_kernel = ndimage.iterate_structure(blackhat_struct, 8)
blackhat_outline = outline + ndimage.black_tophat(outline,

                                structure=blackhat_kernel)



plt.title('Blackhat outline')

plt.imshow(blackhat_outline, cmap='gray');
lung_filter = np.bitwise_or(marker_internal, blackhat_outline)

plt.title('lung filter')

plt.imshow(lung_filter, cmap='gray');
lung_filter = ndimage.morphology.binary_closing(lung_filter,

                structure=np.ones((5,5)), iterations=3)

plt.title('Lung Filter via internal marker and blackhat outline')

plt.imshow(lung_filter, cmap='gray');
#where you see a 1 in lung filter - put the actual pixel value

#everywhere else include -2000

plt.title('Our segmented slice!!')

plt.imshow(np.where(lung_filter == 1, test_slice, -2000), cmap='gray');
def gen_internal_marker(slices_s, threshold= -300):

    internal_marker = slices_s < threshold

    internal_marker_labels = measure.label(segmentation.clear_border(internal_marker))

    areas = [x.area for x in measure.regionprops(internal_marker_labels)]

    areas.sort()

    for region in measure.regionprops(internal_marker_labels):

        if region.area < areas[-2]:

            for coordinates in region.coords:

                internal_marker_labels[coordinates[0], coordinates[1]] = 0

    marker_internal = internal_marker_labels > 0

                

    return marker_internal



def gen_external_marker(internal_marker, iter_1 = 10, iter_2 = 50):

    external_a = ndimage.binary_dilation(internal_marker, 

                                         iterations=iter_1)

    external_b = ndimage.binary_dilation(internal_marker, 

                                         iterations=iter_2)

    external_marker = external_b ^ external_a

    return external_marker



def gen_watershed_marker(internal_marker, external_marker):

    watershed_marker = np.zeros((512, 512), dtype=np.int)

    watershed_marker += internal_marker * 255

    watershed_marker += external_marker * 128

    return watershed_marker



def gen_sobel_grad(one_slice):

    x_edges = ndimage.sobel(one_slice, 1)

    y_edges = ndimage.sobel(one_slice, 0)

    sobel_grad = np.hypot(x_edges, y_edges)

    sobel_grad *= 255.0 / np.max(sobel_grad)

    return sobel_grad



def gen_blackhat_outline(watershed, blackhat_struct, b_hat_iters=1):

    outline = ndimage.morphological_gradient(watershed, size=(3,3))

    blackhat_kernel = ndimage.iterate_structure(blackhat_struct, 

                                               b_hat_iters)

    blackhat_outline = outline + ndimage.black_tophat(outline, 

                            structure=blackhat_kernel)

    return blackhat_outline





def gen_lung_filter(internal_marker, blackhat_outline,

                   kernel_size=(5,5), iterations=3):

    pre_filter = np.bitwise_or(internal_marker, blackhat_outline)

    lung_filter = ndimage.morphology.binary_closing(pre_filter,

                            structure=np.ones(kernel_size),

                            iterations=iterations)

    return lung_filter





def watershed_seg(slice_s, blackhat_struct, threshold=-350,

                  b_hat_iters=1, iter_1=10, iter_2=50):

    

    scan = [] #initialize an empty list

    for one_slice in slice_s:

        internal_marker = gen_internal_marker(one_slice)



        external_marker = gen_external_marker(internal_marker)

        

        watershed_marker = gen_watershed_marker(internal_marker,

                                               external_marker)

        

        sobel_grad = gen_sobel_grad(one_slice)

       

        watershed = segmentation.watershed(sobel_grad, 

                                           watershed_marker)

        

        blackhat_outline = gen_blackhat_outline(watershed,

                                blackhat_struct, b_hat_iters)

        

        lung_filter = gen_lung_filter(internal_marker,

                                      blackhat_outline)

        

        segmented_slice = np.where(lung_filter == 1, one_slice, -2000)

        scan.append(segmented_slice)

        

    return np.array(scan)
#just a reminder of the shape of the middle section of the scan we pulled earlier

middle_slices_hu.shape
%%time

segmented_scan_1 = watershed_seg(middle_slices_hu, blackhat_struct,

                                b_hat_iters=1)
%%time

segmented_scan_6 = watershed_seg(middle_slices_hu, blackhat_struct,

                              b_hat_iters=6)
fig, ax = plt.subplots(1, 2, figsize=(20,8))

ax[0].imshow(segmented_scan_1[32], cmap='Blues_r')

ax[1].imshow(segmented_scan_6[32], cmap='Blues_r');
fig,ax = plt.subplots(12,5, figsize=(20,20))

for n in range(12):

    for m in range(5):

        ax[n,m].imshow(segmented_scan_1[n*5+m], cmap='Blues_r')