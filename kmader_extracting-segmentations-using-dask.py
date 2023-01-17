import numpy
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
map_base_dir = '../input/'
map_img_dir = '../input/train/images/'
json_path = os.path.join(map_base_dir, 'annotation.json')
with open(json_path, 'r') as f:
    annot_data = json.load(f)
image_df = pd.DataFrame(annot_data['images'])
image_df.sample(3)
fig, m_axs = plt.subplots(2, 2, figsize = (10, 10))
for c_ax, (_, c_row) in zip(m_axs.flatten(), image_df.sample(4).iterrows()):
    img_data = imread(os.path.join(map_img_dir, c_row['file_name']))
    c_ax.imshow(img_data)
annot_df = pd.DataFrame(annot_data['annotations'])
annot_df.sample(3)
full_df = pd.merge(annot_df, image_df, how='left', left_on = 'image_id', right_on='id').dropna()
print(image_df.shape[0], '+', annot_df.shape[0], '->', full_df.shape[0])
full_df.sample(2)
def create_boxes(in_rows):
    #TODO: this seems to get a few of the boxes wrong so we stick to segmentation polygons instead
    box_list = []
    for _, in_row in in_rows.iterrows():
        # bbox from the coco standard
        (start_y, start_x, wid_y, wid_x) = in_row['bbox']
        
        box_list += [Rectangle((start_x, start_y), 
                         wid_y , wid_x
                         )]
    return box_list
fig, m_axs = plt.subplots(2, 2, figsize = (10, 10))
for c_ax, (c_id, c_df) in zip(m_axs.flatten(), full_df.groupby('image_id')):
    img_data = imread(os.path.join(map_img_dir, c_df['file_name'].values[0]))
    c_ax.imshow(img_data)
    #c_ax.add_collection(PatchCollection(create_boxes(c_df), alpha = 0.25, facecolor = 'red'))
    for _, c_row in c_df.iterrows():
        xy_vec = np.array(c_row['segmentation']).reshape((-1, 2))
        c_ax.plot(xy_vec[:, 0], xy_vec[:, 1], label = c_df['id_x'])
from matplotlib.path import Path
from skimage.color import label2rgb
def rows_to_segmentation(in_img, in_df):
    xx, yy = np.meshgrid(range(in_img.shape[0]), 
                range(in_img.shape[1]),
               indexing='ij')
    out_img = np.zeros(in_img.shape[:2])
    for _, c_row in in_df.iterrows():
        xy_vec = np.array(c_row['segmentation']).reshape((-1, 2))
        c_ax.plot(xy_vec[:, 0], xy_vec[:, 1], label = c_df['id_x'])
        xy_path = Path(xy_vec)
        out_img += xy_path.contains_points(np.stack([yy.ravel(), 
                                                     xx.ravel()], -1)).reshape(out_img.shape)
    return out_img
fig, m_axs = plt.subplots(3, 3, figsize = (15, 20))
for (c_ax, d_ax, f_ax), (c_id, c_df) in zip(m_axs,
                                      full_df.groupby('image_id')):
    img_data = imread(os.path.join(map_img_dir, c_df['file_name'].values[0]))
    c_ax.imshow(img_data)
    out_img = rows_to_segmentation(img_data, c_df)
    rgba_img = np.concatenate([img_data, 
                               np.clip(np.expand_dims(127*out_img+127, -1), 0, 255).astype(np.uint8)
                              ], -1)
    d_ax.imshow(rgba_img)
    
    f_ax.imshow(label2rgb(image=img_data, label=out_img, bg_label = 0))
from sklearn.model_selection import train_test_split
train_ids, valid_ids = train_test_split(image_df['id'], test_size = 0.25)
train_df = full_df[full_df['image_id'].isin(train_ids)]
valid_df = full_df[full_df['image_id'].isin(valid_ids)]
print(train_df.shape[0], 'training boxes')
print(valid_df.shape[0], 'validation boxes')
def single_img_gen(c_df):
    """
    function to get a single image from a part of a dataframe
    """
    img_data = imread(os.path.join(map_img_dir, c_df['file_name'].values[0]))
    out_seg = np.expand_dims(rows_to_segmentation(img_data, c_df), -1)
    return (img_data/255.0).astype(np.float32), out_seg.astype(np.float32)  
from skimage.util.montage import montage2d
single_df = valid_df[valid_df['image_id'].isin([valid_df['image_id'].values[0]])]
t_x, t_y = single_img_gen(single_df)   
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
montage_rgb = lambda x: np.stack([montage2d(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ax1.imshow(t_x)
ax2.imshow(t_y[:, :, 0], cmap = 'bone_r')
import dask.array as da
import dask
import dask.diagnostics as diag
from multiprocessing.pool import ThreadPool
import h5py
from bokeh.io import output_notebook
from bokeh.resources import CDN
output_notebook(CDN, hide_banner=True)
@dask.delayed
def dgen_as_alpha(in_df):
    x, y = single_img_gen(in_df)
    return np.concatenate([x, y], -1)

def dask_read_seg(in_df, max_items = 1000):
    lazy_images = [dgen_as_alpha(c_df.copy()) for _, (_, c_df) in zip(range(max_items), 
                                                          in_df.groupby('image_id'))
                  ]     # Lazily evaluate on each group
    s_img = lazy_images[0].compute()
    arrays = [da.from_delayed(lazy_image,           # Construct a small Dask array
                              dtype=s_img.dtype,   # for every lazy value
                              shape=s_img.shape)
              for lazy_image in lazy_images]

    return da.stack(arrays, axis=0)                # Stack all small Dask arrays into one
# larger chunks are more efficient for writing/compressing and make the paralellization more efficient
big_chunker = lambda x: x.rechunk({0: x.shape[0]//32, 1: -1, 2: -1, 3: -1})
train_array = big_chunker(dask_read_seg(train_df, 750))
print(train_array)
valid_array = big_chunker(dask_read_seg(valid_df, 250))
print(valid_array)
!rm *.h5 # just make sure there are no files
with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:
    with dask.config.set(pool=ThreadPool(4)):
        train_array.to_hdf5('train.h5', '/images', compression = 'lzf')
        valid_array.to_hdf5('valid.h5', '/images', compression = 'lzf')
!ls -lh *.h5
diag.visualize([prof, rprof])
@dask.delayed
def dgen_just_seg(in_df):
    _, y = single_img_gen(in_df)
    return y.astype(bool) # much smaller

def dask_read_just_seg(in_df, max_items = 1000):
    lazy_images = [(c_id, dgen_just_seg(c_df.copy())) for _, (c_id, c_df) in zip(range(max_items), 
                                                          in_df.groupby('image_id'))
                  ]     # Lazily evaluate on each group
    s_img = lazy_images[0][1].compute()
    arrays = [da.from_delayed(lazy_image,           # Construct a small Dask array
                              dtype=s_img.dtype,   # for every lazy value
                              shape=s_img.shape)
              for _, lazy_image in lazy_images]
    img_ids = [c_id for c_id, _ in lazy_images]

    return img_ids, da.stack(arrays, axis=0)   # Stack all small Dask arrays into one
all_ids, all_array = dask_read_just_seg(full_df, 10000)
print(all_array)
with open('all_segmentations.json', 'w') as f:
    json.dump({'image_id': all_ids}, f)
big_chunker = lambda x: x.rechunk({0: x.shape[0]//128, 1: -1, 2: -1, 3: -1})
all_array = big_chunker(all_array)
all_array
with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:
    with dask.config.set(pool=ThreadPool(4)):
        all_array.to_hdf5('all_segmentations.h5', '/images', 
                          compression = 'lzf')
!ls -lh *.h5
diag.visualize([prof, rprof])