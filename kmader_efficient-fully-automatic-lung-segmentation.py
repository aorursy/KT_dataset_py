import os

from glob import glob

from skimage.io import imread

import nibabel as nib

import matplotlib.pyplot as plt

from skimage.measure import label, regionprops, perimeter

from scipy import ndimage as ndi

from scipy.cluster.vq import kmeans

import numpy as np

p_args = dict(cmap = 'bone', interpolation = 'none')
base_path = os.path.join('..', 'input', '3d_images')

all_images = glob(os.path.join(base_path,'IMG_*'))

all_masks = [c_path.replace('IMG_','MASK_') for c_path in all_images]

print(len(all_masks),' matching files found:',all_masks[0])
img=nib.load(all_images[0])

test_image=img.get_data()



c_mask=nib.load(all_masks[0])

test_mask=c_mask.get_data()

print('loading images:', test_image.shape, 'and mask:', test_mask.shape)

ref_slice_idx = 100
%matplotlib inline    

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(test_image[ref_slice_idx], **p_args)

ax1.set_title('Input Image')

ax2.imshow(test_mask[ref_slice_idx], **p_args)

ax2.set_title('Ground Truth')
from skimage.morphology import reconstruction

def fill_image(in_image):

    seed = np.copy(in_image)

    seed[1:-1, 1:-1] = in_image.max()

    mask = in_image

    return reconstruction(seed, mask, method='erosion')

def fill_image_3d(in_image):

    return np.stack([fill_image(c_slice) for c_slice in in_image],0)
%%time

fl_image = fill_image_3d(test_image)
dfl_image = np.abs(fl_image - test_image)

fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize = (9,3))

ax1.imshow(test_image[ref_slice_idx], **p_args)

ax1.set_title('Input Image')

ax2.imshow(fl_image[ref_slice_idx],**p_args)

ax2.set_title('Filled Image')

ax3.imshow(dfl_image[ref_slice_idx], **p_args)

ax3.set_title('Difference Image')
def kmeans_thresh(in_img):

    centers, _ = kmeans(in_img.ravel().astype(np.float32), 2)

    tissue_center, air_center = sorted(centers)

    return np.abs(in_img-air_center)<np.abs(in_img-tissue_center)
k_seg = kmeans_thresh(dfl_image)

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(test_image[ref_slice_idx], cmap = 'bone')

ax1.set_title('Input Image')

ax2.imshow(k_seg[ref_slice_idx], cmap = 'bone')

ax2.set_title('KMeans Seg')
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize = (9,3))

ax1.imshow(np.sum(test_image,1), cmap = 'bone')

ax1.set_title('Input Image')

ax2.imshow(np.sum(k_seg,1), cmap = 'bone')

ax2.set_title('KMeans Segmented')

ax3.imshow(np.sum(test_mask,1), cmap = 'bone')

ax3.set_title('Ground Truth')
def kmeans_thresh(in_img):

    """

    Use kmeans to automatically find a threshold for a two-class image like tissue/air

    :param in_img:

    :return:

    >>> kmeans_thresh(np.eye(3))

    array([[ True, False, False],

           [False,  True, False],

           [False, False,  True]], dtype=bool)

    """

    centers, _ = kmeans(in_img.ravel().astype(np.float32), 2)

    tissue_center, air_center = sorted(centers)

    return np.abs(in_img-air_center)<np.abs(in_img-tissue_center)
big_seg = take_biggest(k_seg)

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(test_image[ref_slice_idx], cmap = 'bone')

ax1.set_title('Input Image')

ax2.imshow(big_seg[ref_slice_idx], cmap = 'bone')

ax2.set_title('Keep Biggest Component')
def reg_to_dist(in_seg):

    lung_roi = regionprops(in_seg.astype(int))[0]

    (z_min, x_min, y_min, z_max, x_max, y_max) = lung_roi.bbox

    xx, yy = np.meshgrid(range(in_seg.shape[2]), range(in_seg.shape[1]))

    cent_x_dist = np.abs(xx-(x_max+x_min)/2)

    out_image = np.zeros_like(in_seg, dtype = np.float32)

    for i, c_slice in enumerate(in_seg):

        slice_labels = label(c_slice)

        for lab_idx in range(1, slice_labels.max()+1):

            lab_mask = (slice_labels==lab_idx) 

            max_dist = cent_x_dist[lab_mask].max()

            out_image[i] += lab_mask * max_dist

    return out_image
big_seg_dist = reg_to_dist(big_seg)

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(test_image[ref_slice_idx], cmap = 'bone')

ax1.set_title('Input Image')

ax2.imshow(big_seg_dist[ref_slice_idx], cmap = 'bone')

ax2.set_title('Distance from Centerline')
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize = (9,3))

ax1.imshow(np.sum(test_image,1), cmap = 'bone')

ax1.set_title('Input Image')

ax2.imshow(np.max(big_seg_dist*(big_seg_dist>50),1), cmap = 'bone')

ax2.set_title('KMeans Segmented')

ax3.imshow(np.sum(test_mask,1), cmap = 'bone')

ax3.set_title('Ground Truth')
from skimage.morphology import dilation, erosion, disk, greyreconstruct

def dilate_label_func(in_lab, d_ele):

    out_lab = in_lab.copy()

    for i in range(1, np.max(in_lab)+1):

        new_lab = dilation(out_lab==i, selem = d_ele)

        # change the added pixels where the old label was empty (don't write over labels)

        out_lab[(new_lab>0) & (out_lab==0)] = i

    return out_lab

    

def prop_slice(last_slice_lab, next_slice_seg, msg = False):

    next_slice_mask = next_slice_seg>0

    next_slice_cnt = np.sum(next_slice_mask)

    cur_slice_lab = last_slice_lab

    dilat_iter = 0

    # run iteratively while there are still unlabeled pixels

    last_pix_count = np.sum((cur_slice_lab*next_slice_mask)>0)

    

    while last_pix_count<next_slice_cnt:

        cur_slice_lab = cur_slice_lab * next_slice_mask

        cur_slice_lab = dilate_label_func(cur_slice_lab, disk(5)) # dilate the image

        dilat_iter+=1

        next_pix_count = np.sum((cur_slice_lab*next_slice_mask)>0)

        if next_pix_count==last_pix_count:

            if msg: print('Forced Stop', '%2.1f%%' % (last_pix_count/next_slice_cnt*100))

            break

            

        last_pix_count = next_pix_count

        if dilat_iter>50:

            if msg: print('Iteration exceeded', dilat_iter)

            break

    # reapply mask

    return cur_slice_lab * next_slice_mask

    

c_label = label(big_seg[ref_slice_idx])

c_label = c_label * (c_label<3)



next_slice = big_seg[ref_slice_idx+20]

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (9,3))

ax1.imshow(c_label)

ax1.set_title('Current Label')

ax2.imshow(c_label+next_slice)

ax2.set_title('Next Slice')

ax3.imshow(prop_slice(c_label, next_slice, msg = True))

ax3.set_title('Propogated Labels')
from tqdm import tqdm

def prop_slice_image(seg_vol, start_lab, start_slice):

    out_vol = np.zeros_like(seg_vol, dtype = start_lab.dtype)

    slice_cnt = seg_vol.shape[0]

    # propagate labels up

    cur_slice = start_lab

    for i in tqdm(range(start_slice, slice_cnt)):

        cur_slice = prop_slice(cur_slice, seg_vol[i])

        out_vol[i] = cur_slice

    # propagate labels down

    cur_slice = start_lab

    for i in tqdm(range(start_slice, 0, -1)):

        cur_slice = prop_slice(cur_slice, seg_vol[i])

        out_vol[i] = cur_slice

    return out_vol
lab_vol = prop_slice_image(big_seg, c_label, ref_slice_idx)
fig, (ax1,ax2, ax3) = plt.subplots(1,3, figsize = (9,3))

ax1.imshow(np.sum(test_image,1), cmap = 'bone')

ax1.set_title('Input Image')

ax2.imshow(np.max(lab_vol,1), cmap = 'gist_earth')

ax2.set_title('KMeans Segmented')

ax3.imshow(np.sum(test_mask,1), cmap = 'bone')

ax3.set_title('Ground Truth')
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (6,3))

ax1.imshow(c_label)

ax1.set_title('Current Label')

ax2.imshow(lab_vol[150])

ax2.set_title('Next Slice')