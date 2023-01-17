%matplotlib inline
from glob import glob
import os, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import seaborn as sns
from skimage.util.montage import montage2d as montage
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# make the necessary conversion
read_hu = lambda x: imread(x).astype(np.float32)-32768
base_img_dir = '../input/minideeplesion/'
patient_df = pd.read_csv('../input/DL_info.csv')
patient_df['kaggle_dir'] = patient_df.apply(lambda c_row: os.path.join(base_img_dir, 
                                                                        '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row)), 1)

patient_df['kaggle_path'] = patient_df.apply(lambda c_row: os.path.join('{kaggle_dir}'.format(**c_row),
                                                                        '{Key_slice_index:03d}.png'.format(**c_row)), 1)

print('Loaded', patient_df.shape[0], 'cases')
patient_df.sample(3)
patient_df['exists'] = patient_df['kaggle_path'].map(os.path.exists)
patient_df = patient_df[patient_df['exists']].drop('exists', 1)
# extact the bounding boxes
patient_df['bbox'] = patient_df['Bounding_boxes'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1, 4)))
patient_df['norm_loc'] = patient_df['Normalized_lesion_location'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1)))
patient_df['Slice_range'] = patient_df['Slice_range'].map(lambda x: [int(y) for y in x.split(',')])
patient_df['Spacing_mm_px_'] = patient_df['Spacing_mm_px_'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1)))
patient_df['Lesion_diameters_Pixel_'] = patient_df['Lesion_diameters_Pixel_'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1)))
patient_df['Radius_x'] = patient_df.apply(lambda x: x['Lesion_diameters_Pixel_'][0]*x['Spacing_mm_px_'][0], 1)
for i, ax in enumerate('xyz'):
    patient_df[f'{ax}_loc'] = patient_df['norm_loc'].map(lambda x: x[i])
print('Found', patient_df.shape[0], 'patients with images')
patient_df['kaggle_stack'] = patient_df.apply(lambda c_row: [os.path.join('{kaggle_dir}'.format(**c_row),
                                                                        '{:03d}.png'.format(i)) for i in range(c_row['Slice_range'][0],
                                                                                                              c_row['Slice_range'][1]+1)], 1)
patient_df['kaggle_stack'] = patient_df['kaggle_stack'].map(lambda file_list: [file_path for file_path in file_list if os.path.exists(file_path)])
# show the stack size
patient_df['kaggle_stack'].map(len).hist()
from skimage.segmentation import mark_boundaries
apply_softwindow = lambda x: (255*plt.cm.gray(0.5*np.clip((x-50)/350, -1, 1)+0.5)[:, :, :3]).astype(np.uint8)

def create_boxes(in_row):
    box_list = []
    for (start_x, start_y, end_x, end_y) in in_row['bbox']:
        box_list += [Rectangle((start_x, start_y), 
                         np.abs(end_x-start_x),
                         np.abs(end_y-start_y)
                         )]
    return box_list
def create_segmentation(in_img, in_row):
    yy, xx = np.meshgrid(range(in_img.shape[0]),
               range(in_img.shape[1]),
               indexing='ij')
    out_seg = np.zeros_like(in_img)
    for (start_x, start_y, end_x, end_y) in in_row['bbox']:
        c_seg = (xx<end_x) & (xx>start_x) & (yy<end_y) & (yy>start_y)
        out_seg+=c_seg
    return np.clip(out_seg, 0, 1).astype(np.float32)
_, test_row = next(patient_df.sample(1, random_state=0).iterrows())
fig, m_axs = plt.subplots(3, 4, figsize = (25, 14))
[x.axis('off') for x in m_axs.flatten()]
for ax1, c_path in zip(m_axs.flatten(), test_row['kaggle_stack']):
    c_img = read_hu(c_path)
    ax1.imshow(c_img, vmin = -1200, vmax = 600, cmap = 'gray')
    ax1.add_collection(PatchCollection(create_boxes(test_row), alpha = 0.25, facecolor = 'red'))
from skimage.transform import resize
import warnings
def smart_stack(in_list, *args, **kwargs):
    """
    Use the first element to determine the size for all the results and resize the ones that dont match
    """
    base_shape = in_list[0].shape
    return np.stack([x if x.shape==base_shape else resize(x, base_shape, preserve_range=True) for x in in_list], *args, **kwargs)
# utility functions compied from https://github.com/4Quant/pyqae
def _dsum(carr,  # type: np.ndarray
          cax  # type: int
          ):
    # type: (...) -> np.ndarray
    """
    Sums the values along all other axes but the current
    """
    return np.sum(carr, tuple(n for n in range(carr.ndim) if n is not cax))

def get_bbox(in_vol,
             min_val=0):
    # type: (np.ndarray, float) -> List[Tuple[int,int]]
    """
    Calculate a bounding box around an image in every direction
    """
    ax_slice = []
    for i in range(in_vol.ndim):
        c_dim_sum = _dsum(in_vol > min_val, i)
        wh_idx = np.where(c_dim_sum)[0]
        c_sl = sorted(wh_idx)
        if len(wh_idx) == 0:
            ax_slice += [(0, 0)]
        else:
            ax_slice += [(c_sl[0], c_sl[-1] + 1)]
    return ax_slice


def apply_bbox(in_vol,  # type: np.ndarray
               bbox_list,  # type: List[Tuple[int,int]]
               pad_values=False,
               padding_mode='edge'
               ):
    # type: (...) -> np.ndarray
    """
    Apply a bounding box to an image
    """

    if pad_values:
        # TODO test padding
        warnings.warn("Padded apply_bbox not fully tested yet", RuntimeWarning)
        n_pads = []  # type: List[Tuple[int,int]]
        n_bbox = []  # type: List[Tuple[int,int]]
        for dim_idx, ((a, b), dim_size) in enumerate(zip(bbox_list,
                                                         in_vol.shape)):
            a_pad = 0 if a >= 0 else -a
            b_pad = 0 if b < dim_size else b - dim_size + 1
            n_pads += [(a_pad, b_pad)]
            n_bbox += [(a + a_pad, b + a_pad)]  # adjust the box

        while len(n_pads)<len(in_vol.shape):
            n_pads += [(0,0)]
        # update the volume
        in_vol = np.pad(in_vol, n_pads, mode=padding_mode)
        # update the bounding box list
        bbox_list = n_bbox

    return in_vol.__getitem__([slice(a, b, 1) for (a, b) in bbox_list])


def autocrop(in_vol,  # type: np.ndarray
             min_val  # type: double
             ):
    # type (...) -> np.ndarray
    """
    Perform an autocrop on an image by keeping all the points above a value
    """
    return apply_bbox(in_vol, get_bbox(in_vol,
                                       min_val=min_val))
thumb_list = []
img_list = []
seg_list = []
path_list = []
from tqdm import tqdm_notebook
import gc; gc.enable()
save_full_imgs = False # do we keep all of the image
for (_, c_row) in tqdm_notebook(patient_df.sample(300).iterrows()):
    
    c_img = read_hu(c_row['kaggle_path'])
    c_seg = create_segmentation(c_img, c_row).astype(bool)
    c_bbox = get_bbox(c_seg)
    
    c_img_list = [read_hu(c_path) for c_path in c_row['kaggle_stack']]
    
    img_list+=[np.stack(c_img_list,0)]
    
    thumb_list+=[np.stack([apply_bbox(cur_image, c_bbox) 
                         for cur_image in c_img_list],0)]
    if save_full_imgs:
        seg_list+=[np.stack([c_seg 
                             for c_path in c_row['kaggle_stack']],0)]
        path_list+=[c_row['File_name']]
    gc.collect()
fig, m_axs = plt.subplots(5, 5, figsize = (20, 20))
for c_ax, c_stack in zip(m_axs.flatten(), thumb_list):
    c_ax.imshow(montage(c_stack), cmap = 'bone', vmin = -500, vmax = 400)
    c_ax.axis('off')
fig.savefig('many_montage.png', dpi = 300)
all_lesions = smart_stack(thumb_list)
montage_3d = lambda x: montage(np.stack([montage(y) for y in x], 0))
fig, ax1 = plt.subplots(1, 1, figsize = (15, 15))
ax1.imshow(montage_3d(all_lesions), cmap = 'bone', vmin = -500, vmax = 400)
fig.savefig('montage.png', dpi = 300)
import h5py
if save_full_imgs:
    with h5py.File('deeplesion.h5', 'w') as h:
        h.create_dataset('image', data=np.expand_dims(smart_stack(img_list, 0), -1), 
                         compression = 5)    
        h.create_dataset('mask', data=np.expand_dims(smart_stack(seg_list, 0), -1).astype(bool), 
                         compression = 5)    
        h.create_dataset('file_name', data=[x.encode('ascii') for x in path_list], 
                         compression = 0)    
# check the file
!ls -lh *.h5
if save_full_imgs:
    with h5py.File('deeplesion.h5', 'r') as h:
        for k in h.keys():
            print(k, h[k].shape, h[k].dtype, h[k].size/1024**2)