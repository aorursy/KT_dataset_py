# Download the ch2better template image for display

!wget https://github.com/Chaogan-Yan/DPABI/raw/master/Templates/ch2better.nii
"""

    Load and display a subject's spatial map

"""



import numpy as np # linear algebra

import nilearn as nl

import nilearn.plotting as nlplt

import nibabel as nib

import h5py

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

mask_filename = '../input/fmri-mask/fMRI_mask.nii'

subject_filename = '../input/trends-assessment-prediction/fMRI_train/10004.mat'

smri_filename = 'ch2better.nii'

mask_niimg = nl.image.load_img(mask_filename)



def load_subject(filename, mask_niimg):

    """

    Load a subject saved in .mat format with

        the version 7.3 flag. Return the subject

        niimg, using a mask niimg as a template

        for nifti headers.

        

    Args:

        filename    <str>            the .mat filename for the subject data

        mask_niimg  niimg object     the mask niimg object used for nifti headers

    """

    subject_data = None

    with h5py.File(subject_filename, 'r') as f:

        subject_data = f['SM_feature'][()]

    # It's necessary to reorient the axes, since h5py flips axis order

    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])

    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)

    return subject_niimg

subject_niimg = load_subject(subject_filename, mask_niimg)

print("Image shape is %s" % (str(subject_niimg.shape)))

num_components = subject_niimg.shape[-1]

print("Detected {num_components} spatial maps".format(num_components=num_components))

nlplt.plot_prob_atlas(subject_niimg, bg_img=smri_filename, view_type='filled_contours', draw_cross=False, title='All %d spatial maps' % num_components, threshold='auto')
grid_size = int(np.ceil(np.sqrt(num_components)))

fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*10, grid_size*10))

[axi.set_axis_off() for axi in axes.ravel()]

row = -1

for i, cur_img in enumerate(nl.image.iter_img(subject_niimg)):

    col = i % grid_size

    if col == 0:

        row += 1

    nlplt.plot_stat_map(cur_img, bg_img=smri_filename, title="IC %d" % i, axes=axes[row, col], threshold=3, colorbar=False)