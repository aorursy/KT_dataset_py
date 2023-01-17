import os

import h5py

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['image.cmap'] = 'viridis'



from skimage import color

from skimage.segmentation import slic
def superpixel_seg(spx, pet_proj, thresh_value = 1.1):

    spx_rgb = color.label2rgb(spx, pet_proj, kind='avg')

    threshold_proj = pet_proj > thresh_value

    all_proj = np.stack((spx_rgb, threshold_proj))

    bright_segs = np.zeros_like(spx)



    for i in np.unique(spx):

        region_mask = (spx == i)

        region_pet = pet_proj[region_mask]

        if region_pet.max() > thresh_value:

            bright_segs[spx == i] = spx[spx == i]

    return bright_segs
with h5py.File(os.path.join('..', 'input', 'lab_petct_vox_5.00mm.h5'), 'r') as p_data:

    ct_images = p_data['ct_data'].items()

    pet_images = p_data['pet_data'].values()

    lab_images = p_data['label_data'].values()

    fig, sb_mat = plt.subplots(7, 4, figsize=(10, 25))

    (ax1s, ax2s, ax3s, ax4s) = sb_mat.T

    for c_ax1, c_ax2, c_ax3, c_ax4, (p_id, ct_img), pet_img, lab_img in zip(ax1s, ax2s, ax3s, ax4s, ct_images, pet_images, lab_images):

        

        ct_image = np.mean(ct_img, 1)[::-1]

        c_ax1.imshow(ct_image, cmap = 'bone')

        c_ax1.set_title('CT')

        c_ax1.axis('off')

        

        pet_proj = np.max(pet_img, 1)[::-1]

        pet_image = np.sqrt(np.max(pet_img, 1).squeeze()[::-1,:])

        c_ax2.imshow(pet_image)

        c_ax2.set_title('PET image')

        c_ax2.axis('off')

        

        spx = slic(pet_proj, n_segments=1000, compactness=0.1)

        b = superpixel_seg(spx + 1, pet_proj, thresh_value=4.1)

        c_ax3.imshow(b)

        c_ax3.set_title('Superpixel segmentation')

        c_ax3.axis('off')

        

        c_ax4.imshow(np.mean(lab_img, 1)[::-1])

        c_ax4.set_title('Label')

        c_ax4.axis('off')