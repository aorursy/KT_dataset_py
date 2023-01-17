!conda install -c conda-forge gdcm -y
!pip install fastai==2.0.8
import numpy as np

import pandas as pd

import os

import pydicom as dcm

import gdcm

from fastai.basics           import *

from fastai.medical.imaging  import *
DICOM_DIR = '../input/osic-pulmonary-fibrosis-progression/train/'

PNG_DIR = '../input/pulmonary-fibrosis-progression-png-training-set/train_png/'
from collections.abc import Iterable



def get_sorted_patient_files(directory):

    """

    Returns all files patients and their files in numerical order.



    Parameters:

    directory                    (str) : Parent directory containing patients



    Returns:

    patient_files   (nested file list) : Outer list has patients and inner lists have their file paths in order

    """

    sorted_files = []

    all_files = [[os.path.join(directory,d ,fn) for fn in os.listdir(directory + d)] for d in os.listdir(directory)]

    for patient_files in all_files: 

        patient_numbers = [int(os.path.basename(fn)[:-4]) for fn in patient_files]

        zipped_list = zip(patient_numbers, patient_files)

        zipped_list = sorted(zipped_list)

        tuples = zip(*zipped_list)

        patient_numbers, patient_files = [list(_tuple) for _tuple in tuples]

        sorted_files.append(patient_files)

    return sorted_files
dicom_files = get_sorted_patient_files(DICOM_DIR)



dicom_files[0][0]
# This originates from: https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai

def fix_pxrepr(dcm):

    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000



def dcm_tfm(fn, im_size=512): 

    try:

        x = dcm.dcmread(fn)

        fix_pxrepr(x)

    except Exception as e:

        print(e)

    if x.Rows != im_size or x.Columns != im_size: x.zoom_to((im_size,im_size))



    px = x.scaled_px

    return TensorImage(px.to_3chan(dicom_windows.lungs,dicom_windows.subdural, bins=None))



def tensor_to_numpy(tensor):

    img = tensor.cpu().numpy()

    return img.transpose(1,2,0)



def open_dicom_normalized(fn, im_size=512):

    """

    This function returns normalized numpy image representation from a dicom path.

    Image channels are: lung window, subdural window, normalized total range

    """

    img = tensor_to_numpy(dcm_tfm(fn, im_size))

    return (img*255).astype(np.uint8)
patient_index = 10

slice_index = 10



_, axs = plt.subplots(1,4,figsize=(16,4))



channels = ['Lung window', 'Subdural window','Normalized image']

for i, (ch, ax) in enumerate(zip(channels,axs.ravel())):

    ax.imshow(open_dicom_normalized(dicom_files[patient_index][slice_index])[:,:,i], cmap='bone')

    ax.set_title(ch)



axs[-1].imshow(open_dicom_normalized(dicom_files[patient_index][slice_index]))

axs[-1].set_title('All three combined')

plt.show()
png_files = get_sorted_patient_files(PNG_DIR)

print(f'Found {len(png_files)} patients')
patient_range = list(range(30,34))

slice_range = list(range(10,50,10))



_, axs = plt.subplots(

    len(slice_range),

    len(patient_range),

    figsize=(len(slice_range)*4,len(patient_range)*4))



for col, patient_index in enumerate(patient_range):

    patient_id = os.path.basename(os.path.dirname(png_files[patient_index][0]))

    axs[0,col].set_title(patient_id)

    

    for row, slice_index in enumerate(slice_range):

        if slice_index < len(png_files[patient_index]):

            axs[row,col].imshow(plt.imread(png_files[patient_index][slice_index]))

        if col==0: axs[row,col].set_ylabel(f'Slice index {slice_index}')