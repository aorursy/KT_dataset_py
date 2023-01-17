

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

# io related

from skimage.io import imread

import os

from glob import glob

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 
all_paths = glob(os.path.join('..', 'input', '*', '*', '*'))

print('Found', len(all_paths), 'images')
from collections import defaultdict

extract_type = lambda full_path: full_path.split(os.path.sep)[2]

extract_key = lambda full_path: os.path.splitext(os.path.basename(full_path))[0].replace('-anno','')

image_dict = defaultdict(dict)

for c_path in all_paths:

    image_dict[extract_key(c_path)][extract_type(c_path)] = c_path

image_df = pd.DataFrame(list(image_dict.values()))

full_count = image_df.shape[0]

image_df.dropna(inplace = True)

print('Total Cases:', full_count, '; Complete Cases:', image_df.shape[0])



image_df.sample(3)
# Output the organized file table

image_df.to_csv('raw_image_paths.csv') # save the paths for use later

image_df.applymap(lambda x: x.replace('../input', '../input/interactive-segmentation-overview')).to_csv('corrected_image_paths.csv')
sample_images = 4

fig, n_axs = plt.subplots(sample_images, 3, figsize = (12, 4*sample_images))

[c_ax.axis('off') for c_ax in n_axs.flatten()] # hide all the ugly axes

for (ax_img, ax_stroke, ax_seg), (_, c_row) in zip(n_axs, 

                                                   image_df.applymap(imread).sample(sample_images).iterrows()):

    ax_img.imshow(c_row['images'])

    # create stroke overlap

    stroke_overlap_image = np.clip(c_row['images']*0.5+c_row['strokes']*1.0, 0, 255).astype(np.uint8)

    ax_stroke.imshow(stroke_overlap_image)

    ax_seg.imshow(c_row['segmentation'], cmap = 'bone')
from scipy.ndimage.morphology import distance_transform_edt

def baseline_seg(in_image, stroke_map):

    # we just use the stroke map for the baseline segmentation

    r_vs_gb_img = ((stroke_map[:,:,0] - 0.5*(stroke_map[:,:,1]+stroke_map[:,:,2]))//100).astype(int)

    dist_map, idx_map = distance_transform_edt(r_vs_gb_img<1, return_indices = True)

    return (r_vs_gb_img[idx_map[0], idx_map[1]])==1
sample_images = 4

fig, n_axs = plt.subplots(sample_images, 3, figsize = (12, 4*sample_images))

[c_ax.axis('off') for c_ax in n_axs.flatten()] # hide all the ugly axes

for (ax_stroke, ax_dilate, ax_seg), (_, c_row) in zip(n_axs, 

                                                   image_df.applymap(imread).sample(sample_images).iterrows()):

    # create stroke overlap

    stroke_overlap_image = np.clip(c_row['images']*0.7+c_row['strokes']*1.0, 0, 255).astype(np.uint8)

    ax_stroke.imshow(stroke_overlap_image)

    ax_stroke.set_title('Image with Strokes')

    # a simple segmentation using the distance transform from the labeled lines

    simple_seg = baseline_seg(c_row['images'], c_row['strokes'])

    ax_dilate.imshow(simple_seg, cmap = 'bone')

    ax_dilate.set_title('Simple Model')

    ax_seg.imshow(c_row['segmentation'], cmap = 'bone')

    ax_seg.set_title('Actual Segmentation')
%%time

full_data_df = image_df.applymap(imread)



full_data_df['baseline_seg'] = full_data_df.apply(lambda c_row: 

                                                  # wrap in a list since pandas gets upset if it's a numpy array

                                                  [baseline_seg(c_row['images'], c_row['strokes'])]

                                                  ,1)
from sklearn.metrics import accuracy_score

full_data_df['accuracy'] = full_data_df.apply(lambda c_row: accuracy_score(c_row['segmentation'].ravel(), 

                                                                          c_row['baseline_seg'][0].ravel()),1)

print('Average Accuracy:', '%2.2f%%' % (100*full_data_df['accuracy'].mean()))

full_data_df['accuracy'].hist()
full_data_df['mse'] = full_data_df.apply(lambda c_row: np.mean(np.power(c_row['segmentation'].ravel().clip(0,1)-

                                                                        c_row['baseline_seg'][0].ravel().clip(0,1),2)),1)

print('Mean Squared Error:', '%2.3f' % (full_data_df['mse'].mean()))

full_data_df['mae'] = full_data_df.apply(lambda c_row: np.mean(np.abs(c_row['segmentation'].ravel().clip(0,1)-

                                                                      c_row['baseline_seg'][0].ravel().clip(0,1))),1)

print('Mean Absolute Error:', '%2.3f' % (full_data_df['mae'].mean()))

full_data_df[['mse', 'mae']].hist()
import h5py

from tqdm import tqdm



def write_df_as_hdf(out_path,

                    out_df,

                    compression='gzip'):

    with h5py.File(out_path, 'w') as h:

        for k, arr_dict in tqdm(out_df.to_dict().items()):

            try:

                s_data = np.stack(arr_dict.values(), 0)

                try:

                    h.create_dataset(k, data=s_data, compression=

                    compression)

                except TypeError as e:

                    try:

                        h.create_dataset(k, data=s_data.astype(np.string_),

                                         compression=compression)

                    except TypeError as e2:

                        print('%s could not be added to hdf5, %s' % (

                            k, repr(e), repr(e2)))

            except ValueError as e:

                print('%s could not be created, %s' % (k, repr(e)))

                all_shape = [np.shape(x) for x in arr_dict.values()]

                warn('Input shapes: {}'.format(all_shape))

from skimage.transform import resize

resize_dim = (256, 256)

def resize_func(in_img):

    if len(in_img.shape)==3:

        return resize(in_img, resize_dim + (in_img.shape[2],), 

                      order = 1, 

                      mode = 'constant')

    elif len(in_img.shape)==2:

        return np.expand_dims(resize(in_img, 

                                     resize_dim, 

                                     order = 1, 

                                     mode = 'constant'),-1)

    else: 

        raise ValueError('Invalid dimension:', in_img.shape)

# prepare output data        

out_data_df = full_data_df[['images','strokes','segmentation']].copy()

def stroke_rgb_to_nd(stroke_rgb):

    stroke_int = ((stroke_rgb[:,:,0] - 0.5*(stroke_rgb[:,:,1]+stroke_rgb[:,:,2]))//100).astype(int)

    return np.stack([stroke_int==1, stroke_int==2], -1) 

out_data_df['strokes'] = out_data_df['strokes'].map(stroke_rgb_to_nd)



write_df_as_hdf('training_data.h5',

                out_data_df.applymap(resize_func))
import h5py

from skimage.util.montage import montage2d

with h5py.File('training_data.h5', 'r') as h5_data:

    for c_key in h5_data.keys():

        print(c_key, h5_data[c_key].shape)

    seg_montage = montage2d(h5_data['segmentation'][:,:,:,0])

    stroke_montage = 2*montage2d(h5_data['strokes'][:,:,:,0])+montage2d(h5_data['strokes'][:,:,:,1])



fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

ax1.imshow(seg_montage, cmap = 'bone')

ax1.set_title('Segmentations')

ax1.axis('off')



ax2.imshow(stroke_montage, cmap = 'gist_earth')

ax2.set_title('Strokes')

ax2.axis('off')



fig.savefig('high_res_output.png', dpi = 300)