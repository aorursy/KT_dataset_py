# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import h5py 

import os

import nilearn as nl

import nilearn.plotting as nlplt

import nibabel as nib

from nilearn import image

from nilearn import plotting

from nilearn import datasets

from nilearn import surface

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data = h5py.File('/kaggle/input/trends-assessment-prediction/fMRI_train/10001.mat', 'r')
def plot_fmri(data_file):

    patient_data = data_file['SM_feature'][:]

    subject_data = np.moveaxis(patient_data, [0,1,2,3], [3,2,1,0])

    fmri_mask = '../input/trends-assessment-prediction/fMRI_mask.nii'

    mask_img = nl.image.load_img(fmri_mask)

    subject_img = nl.image.new_img_like(mask_img, subject_data, affine=mask_img.affine, copy_header=True)

    first_rsn = image.index_img(subject_img, 0)

    print(first_rsn.shape)

    plotting.plot_stat_map(first_rsn)
directory = '/kaggle/input/trends-assessment-prediction/fMRI_train'

five = 0

for file in os.listdir(directory):

    if five > 1:

        break

    if file.endswith(".mat"): 

        data = h5py.File(os.path.join(directory, file), 'r')

        print(file)

        five += 1

        plot_fmri(data)
first_rsn = image.index_img(subject_img, 0)

print(first_rsn.shape)

plotting.plot_stat_map(first_rsn)