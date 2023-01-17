# !conda install -c conda-forge gdcm -y # run this code for the first time
import os
# import gdcm
from collections import defaultdict
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import figure_factory as FF

import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import random
import pydicom
# load train and test image

DICOM_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/train'
DICOM_DIR_TEST = '/kaggle/input/osic-pulmonary-fibrosis-progression/test'

dicom_dict = defaultdict(list)
dicom_dict_test = defaultdict(list)

default_image_size = 512

for dirname in os.listdir(DICOM_DIR):
    path = os.path.join(DICOM_DIR, dirname)
    dicom_dict[dirname].append(path)
    
for dirname in os.listdir(DICOM_DIR_TEST):
    path = os.path.join(DICOM_DIR_TEST, dirname)
    dicom_dict_test[dirname].append(path)

# ### load_scan:
# since there are a couple of dicom files that don't have 'ImagePositionPatient' attribute so instead,
# i will use 'InstanceNumber' attribute for those

# ### dicom_file:
# 1. take index number of patient which stored in dict_dicom earlier
# 2. this might be useful when you need to pick some random patient
# 3. It also takes specific patient Id in case you need.
# 4. Note that this function is going to read all file in taken path.

# ### get_pixels_hu
# 1. take dicom file which had called through dicom_file function
# 2. It stacks up all the load slices of certain patient
# 3. stacked slices will be calculated into Hounsfield Units



def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    length = len(slices)
    actualLength = sum([1 if hasattr(x, 'ImagePositionPatient') else 0 for x in slices])
    if length == actualLength:
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    else:
        slices.sort(key = lambda x: float(x.InstanceNumber))
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image < -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slbice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
test = load_scan(dicom_dict['ID00047637202184938901501'][0])

test_hu = get_pixels_hu(test)
print('Patient {}'.format(test[0].PatientName))
print('Slices : {}\nPixels : ({} x {})'.format(test_hu.shape[0], test_hu.shape[1], test_hu.shape[2]))

plt.figure(figsize=(12, 8))
ax = sns.distplot(test_hu.flatten(), bins=80, norm_hist=True)
# ax.set_title('Hounsfield Units of patient {}'.format(test_hu[0].PatientName), fontsize=25)
plt.show()
# ref) https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
# INTER_NEAREST - a nearest-neighbor interpolation
# INTER_LINEAR - a bilinear interpolation (used by default)
# INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
# INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
# INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

import cv2
def resize(slices):
    new_slice = []
    for slice_number in range(len(slices)):
        new_slice.append(cv2.resize(slices[slice_number], dsize=(default_image_size, default_image_size), interpolation=cv2.INTER_AREA))
    return np.array(new_slice)
# # do crop here
def crop(slices):
    slice_height = len(slices[0])
    slice_width = len(slices[0][0])
    diff_height_half = int((slice_height - default_image_size) / 2)
    diff_width_half = int((slice_width - default_image_size) / 2)
    new_slice = []
    for slice_number in range(len(slices)):
        new_slice.append(slices[slice_number][diff_width_half: slice_width - diff_width_half, diff_height_half: slice_height - diff_height_half])
    return np.array(new_slice)
def resample(image, slice_tickness, x_pixel_spacing, y_pixel_spacing, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([slice_tickness, x_pixel_spacing, y_pixel_spacing], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing
def make_mesh(image, threshold):
    p = image.transpose(2, 1, 0)
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    return verts, faces

def static_3d(image, threshold=-300):
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    verts, faces = make_mesh(image, threshold)
    x, y, z = zip(*verts)
    
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    
    ax.add_collection3d(mesh)
    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    plt.show()
    
def interactive_3d(image, threshold=-300):
    verts, faces = make_mesh(image, threshold)
    x, y, z = zip(*verts)
    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            simplices=faces)
    iplot(fig)
x_size = test[0].Columns
y_size = test[0].Rows
slice_thinkness = test[0].SliceThickness
x_pixel_spacing = test[0].PixelSpacing[0] * (x_size / default_image_size)
y_pixel_spacing = test[0].PixelSpacing[1] * (y_size / default_image_size)

# if ratio is not 1:1 then crop
if (x_size != y_size):
    test_hu = crop(test_hu)

# if size is not 512 then resize
if (x_size != default_image_size or y_size != default_image_size):
    test_hu = resize(test_hu)

# resample test
resampled_test_hu, spacing = resample(test_hu, slice_thinkness, x_pixel_spacing, y_pixel_spacing)
static_3d(resampled_test_hu)
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
    z, y, x = labels.shape
    for slice_number in range(len(image)):
        #Fill the air around the person
        binary_image[slice_number][labels[slice_number, 0, 0] == labels[slice_number]] = 2
        binary_image[slice_number][labels[slice_number, 0, x - 1] == labels[slice_number]] = 2
        binary_image[slice_number][labels[slice_number, y - 1, 0] == labels[slice_number]] = 2
        binary_image[slice_number][labels[slice_number, y - 1, x - 1] == labels[slice_number]] = 2
        binary_image[slice_number][labels[slice_number, 0, int((x - 1) / 2)] == labels[slice_number]] = 2
        binary_image[slice_number][labels[slice_number, y - 1, int((x - 1) / 2)] == labels[slice_number]] = 2
    
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
    return binary_image
segmented_lungs = segment_lung_mask(resampled_test_hu, False)
segmented_lungs_fill = segment_lung_mask(resampled_test_hu, True)
static_3d(segmented_lungs, 1.5)
static_3d(segmented_lungs_fill, 1.5)
static_3d(segmented_lungs_fill - segmented_lungs, -0.5)
interactive_3d(segmented_lungs_fill - segmented_lungs, -0.5)
# create directories
# os.removedirs("/kaggle/working/train")
# os.removedirs("/kaggle/working/test")
# os.makedirs('train')
# os.makedirs('test')
# load all dicom
# import gdcm

# PATH = '/kaggle/working/train'
# PATH_TEST = '/kaggle/working/test'

# # create train data npy
# for k, v in dicom_dict.items():
#     print(k ,v)
#     scan = load_scan(v[0])
#     scan_hu = get_pixels_hu(scan)
#     resample_scan_hu, spacing = resample(scan_hu, scan)
#     segmented_lungs = segment_lung_mask(resample_scan_hu, False)
#     segmented_lungs_fill = segment_lung_mask(resample_scan_hu, True)
#     np.save(f'{PATH}/{k}' , segmented_lungs_fill - segmented_lungs)
    
# create test data npy
# for k, v in dicom_dict_test.items():
#     print(k ,v)
#     scan = load_scan(v[0])
#     scan_hu = get_pixels_hu(scan)
#     resample_scan_hu, spacing = resample(scan_hu, scan)
#     segmented_lungs = segment_lung_mask(resample_scan_hu, False)
#     segmented_lungs_fill = segment_lung_mask(resample_scan_hu, True)
#     np.save(f'{PATH_TEST}/{k}' , segmented_lungs_fill - segmented_lungs)