import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util.montage import montage2d as montage
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')
import gc; gc.enable() # memory is tight

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
masks = pd.read_csv(os.path.join('../input/',
                                 'train_ship_segmentations.csv'))
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])
import dask.array as da
import dask
import dask.diagnostics as diag
from multiprocessing.pool import ThreadPool
import h5py
from bokeh.io import output_notebook
from bokeh.resources import CDN
output_notebook(CDN, hide_banner=True)
all_batches = list(masks.groupby('ImageId'))
all_ids = pd.DataFrame({'ImageId': [id for id, _ in all_batches]})
all_ids.to_csv('image_ids.csv', index=False)
all_ids.sample(2)
def dask_read_seg(in_batches, max_items = None):
    d_mask_fun = dask.delayed(masks_as_image)
    if max_items is None:
        max_items = len(in_batches)
    lazy_images = [d_mask_fun(c_masks['EncodedPixels'].values) 
                   for _, (_, c_masks) in zip(range(max_items), in_batches)
                  ]     # Lazily evaluate on each group
    s_img = lazy_images[0].compute()
    arrays = [da.from_delayed(lazy_image,           # Construct a small Dask array
                              dtype=s_img.dtype,   # for every lazy value
                              shape=s_img.shape)
              for lazy_image in lazy_images]

    return da.stack(arrays, axis=0)                # Stack all small Dask arrays into one
tiny_img_ds = dask_read_seg(all_batches, 20)
tiny_img_ds
with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:
    with dask.config.set(pool=ThreadPool(4)):
        tiny_img_ds.to_hdf5('tiny_segmentions.h5', '/image', compression = 'lzf')
!ls -lh *.h5
# larger chunks are more efficient for writing/compressing and make the paralellization more efficient
larger_chunker = lambda x: x.rechunk({0: x.shape[0]//400, 1: -1, 2: -1, 3: -1})
all_img_ds = larger_chunker(dask_read_seg(all_batches))
all_img_ds
with diag.ProgressBar(), diag.Profiler() as prof, diag.ResourceProfiler(0.5) as rprof:
    with dask.config.set(pool=ThreadPool(4)):
        all_img_ds.to_hdf5('segmentions.h5', '/image', compression = 'lzf')
!ls -lh *.h5
