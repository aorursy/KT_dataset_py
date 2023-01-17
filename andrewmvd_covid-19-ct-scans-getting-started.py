import glob

import pandas  as pd

import numpy   as np

import nibabel as nib

import matplotlib.pyplot as plt
# Read and examine metadata

raw_data = pd.read_csv('../input/covid19-ct-scans/metadata.csv')

raw_data.sample(5)
def read_nii(filepath):

    '''

    Reads .nii file and returns pixel array

    '''

    ct_scan = nib.load(filepath)

    array   = ct_scan.get_fdata()

    array   = np.rot90(np.array(array))

    return(array)
# Read sample

sample_ct   = read_nii(raw_data.loc[0,'ct_scan'])

sample_lung = read_nii(raw_data.loc[0,'lung_mask'])

sample_infe = read_nii(raw_data.loc[0,'infection_mask'])

sample_all  = read_nii(raw_data.loc[0,'lung_and_infection_mask'])
# Examine Shape

sample_ct.shape
def plot_sample(array_list, color_map = 'nipy_spectral'):

    '''

    Plots and a slice with all available annotations

    '''

    fig = plt.figure(figsize=(18,15))



    plt.subplot(1,4,1)

    plt.imshow(array_list[0], cmap='bone')

    plt.title('Original Image')



    plt.subplot(1,4,2)

    plt.imshow(array_list[0], cmap='bone')

    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)

    plt.title('Lung Mask')



    plt.subplot(1,4,3)

    plt.imshow(array_list[0], cmap='bone')

    plt.imshow(array_list[2], alpha=0.5, cmap=color_map)

    plt.title('Infection Mask')



    plt.subplot(1,4,4)

    plt.imshow(array_list[0], cmap='bone')

    plt.imshow(array_list[3], alpha=0.5, cmap=color_map)

    plt.title('Lung and Infection Mask')



    plt.show()
# Examine one slice of a ct scan and its annotations

plot_sample([sample_ct[...,120], sample_lung[...,120], sample_infe[...,120], sample_all[...,120]])
def bulk_plot_sample(array_list, index_list):

    '''

    Plots multiple slices, wrapper of plot_sample

    '''

    for index_value in index_list:

        plot_sample([array_list[0][...,index_value], array_list[1][...,index_value], array_list[2][...,index_value], array_list[3][...,index_value]])
# Examine multiple slices of a ct scan and its annotations

bulk_plot_sample([sample_ct, sample_lung, sample_infe, sample_all], index_list=[100,110,120,130,140,150])