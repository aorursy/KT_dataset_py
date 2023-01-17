import os

import h5py

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



from skimage.segmentation import slic

from skimage.segmentation import mark_boundaries



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
ct_images = []

pet_images = []

label_images = []

with h5py.File(os.path.join('..', 'input', 'lab_petct_vox_5.00mm.h5'), 'r') as p_data:

    print('Available keys:', list(p_data.keys()))

    print('Available patients:', len(p_data['ct_data']))

    id_list = list(p_data['ct_data'].keys())

    for i in range(len(id_list)):

        ct_images.append(p_data['ct_data'][id_list[i]].value)

        pet_images.append(p_data['pet_data'][id_list[i]].value)

        label_images.append(p_data['label_data'][id_list[i]].value)
def label_score(gt_labels, sp_segs):

    # type: (np.ndarray, np.ndarray) -> float

    """

    Score how well the superpixels match to the ground truth labels. 

    Here we use a simple penalty of number of pixels misclassified

    :param gt_labels: the ground truth labels (from an annotation tool)

    :param sp_segs: the superpixel segmentation

    :return: the score (lower is better)

    """

    out_score = 0

    for idx in np.unique(sp_segs):

        cur_region_mask = sp_segs == idx

        labels_in_region = gt_labels[cur_region_mask]

        if np.sum(labels_in_region) > 0:

            out_score += np.sum(pd.value_counts(labels_in_region)[1:].values)

    return out_score
# Make new superpixels

def make_superpixel(ct_image,

                    pet_image,  

                    label_image,

                    pet_weight = 1.0, # how strongly to weight the pet_signal (1.0 is the same as CT)

                    n_segments = 1000, # number of segments

                    compactness = 1): # how compact the segments are

    

    t_petct_vol = np.stack([np.stack([(ct_slice+1024).clip(0,2048)/2048, 

                            pet_weight*(suv_slice).clip(0,5)/5.0

                           ],-1) for ct_slice, suv_slice in zip(ct_image, pet_image)],0)

    petct_segs = slic(t_petct_vol, 

                      n_segments=n_segments, 

                      compactness=compactness,

                      multichannel=True)

    return petct_segs



def make_and_score(*args, **kwargs):

    n_segs = make_superpixel(*args, **kwargs)

    return label_score(label_image, n_segs)
# Optimize the values

from scipy.optimize import fmin



def f_make(n, *args):

    """A wrapper because fmin inputs needs to be floats for optimization. """

    print('calling f_make with:', n)

    return make_and_score(n_segments=int(n*1000), *args)



for ct_image, pet_image, label_image in zip(ct_images, pet_images, label_images):

    fmin(f_make, x0=[1], args=(ct_image, pet_image, label_image))

    break