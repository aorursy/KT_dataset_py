# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import SimpleITK as sitk

from glob import glob

from skimage.util import montage as montage2d

import matplotlib.pyplot as plt

def safe_sitk_read(folder_name, *args, **kwargs):

    """

    Since the default function just looks at images 0 and 1 to determine slice thickness

    and the images are often not correctly alphabetically sorted

    :param folder_name: folder to read

    :return:

    """

    dicom_names = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(folder_name)

    return sitk.ReadImage(dicom_names, *args, **kwargs)

def sitk_to_np(in_img):

    # type: (sitk.Image) -> Tuple[np.ndarray, Tuple[float, float, float]]

    return sitk.GetArrayFromImage(in_img), in_img.GetSpacing()
patient_folders = glob('../input/second-annual-data-science-bowl/train/train/1/study/*')

print(patient_folders)

first_pat = safe_sitk_read(patient_folders[0])

pat_img, pat_spc = sitk_to_np(first_pat)
pat_spc
%matplotlib inline

plt.imshow(montage2d(pat_img), cmap = 'bone')