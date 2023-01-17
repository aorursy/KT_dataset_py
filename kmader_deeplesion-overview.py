%matplotlib inline
from glob import glob
import os, pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# make the necessary conversion
read_hu = lambda x: imread(x).astype(np.float32)-32768
base_img_dir = '../input/minideeplesion/'
patient_df = pd.read_csv('../input/DL_info.csv')
patient_df['kaggle_path'] = patient_df.apply(lambda c_row: os.path.join(base_img_dir, 
                                                                        '{Patient_index:06d}_{Study_index:02d}_{Series_ID:02d}'.format(**c_row),
                                                                        '{Key_slice_index:03d}.png'.format(**c_row)), 1)
patient_df['Radius'] = patient_df['Lesion_diameters_Pixel_'].map(lambda x: float(x.split(', ')[0]))
print('Loaded', patient_df.shape[0], 'cases')
patient_df.sample(3)
sns.pairplot(hue='Patient_gender', data=patient_df[['Patient_age', 'Patient_gender', 'Key_slice_index', 'Radius']])
patient_df['exists'] = patient_df['kaggle_path'].map(os.path.exists)
patient_df = patient_df[patient_df['exists']].drop('exists', 1)
# extact the bounding boxes
patient_df['bbox'] = patient_df['Bounding_boxes'].map(lambda x: np.reshape([float(y) for y in x.split(',')], (-1, 4)))
print('Found', patient_df.shape[0], 'patients with images')
def create_boxes(in_row):
    box_list = []
    for (start_x, start_y, end_x, end_y) in in_row['bbox']:
        box_list += [Rectangle((start_x, start_y), 
                         np.abs(end_x-start_x),
                         np.abs(end_y-start_y)
                         )]
    return box_list
_, test_row = next(patient_df.sample(1, random_state=0).iterrows())
fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
c_img = read_hu(test_row['kaggle_path'])
ax1.imshow(c_img, vmin = -1200, vmax = 600, cmap = 'gray')
ax1.add_collection(PatchCollection(create_boxes(test_row), alpha = 0.25, facecolor = 'red'))
ax1.set_title('{Patient_age}-{Patient_gender}'.format(**test_row))
fig, m_axs = plt.subplots(3, 3, figsize = (20, 20))
for c_ax, (_, c_row) in zip(m_axs.flatten(), 
        patient_df.sample(50, random_state=0).iterrows()):
    
    c_img = read_hu(c_row['kaggle_path'])
    c_ax.imshow(c_img, vmin = -1200, vmax = 600, cmap = 'gray')
    c_ax.add_collection(PatchCollection(create_boxes(c_row), alpha = 0.25, facecolor = 'red'))
    c_ax.set_title('{Patient_age}-{Patient_gender}'.format(**c_row))
    c_ax.axis('off')
fig.savefig('overview.png', dpi = 300)
def create_segmentation(in_img, in_row):
    yy, xx = np.meshgrid(range(in_img.shape[0]),
               range(in_img.shape[1]),
               indexing='ij')
    out_seg = np.zeros_like(in_img)
    for (start_x, start_y, end_x, end_y) in in_row['bbox']:
        c_seg = (xx<end_x) & (xx>start_x) & (yy<end_y) & (yy>start_y)
        out_seg+=c_seg
    return np.clip(out_seg, 0, 1).astype(np.float32)
from skimage.segmentation import mark_boundaries
apply_softwindow = lambda x: (255*plt.cm.gray(0.5*np.clip((x-50)/350, -1, 1)+0.5)[:, :, :3]).astype(np.uint8)
fig, m_axs = plt.subplots(3, 2, figsize = (10, 15))
for (ax1, ax2), (_, c_row) in zip(m_axs, 
        patient_df.sample(50, random_state=0).iterrows()):
    
    c_img = read_hu(c_row['kaggle_path'])
    ax1.imshow(c_img, vmin = -1200, vmax = 600, cmap = 'gray')
    ax1.add_collection(PatchCollection(create_boxes(c_row), alpha = 0.25, facecolor = 'red'))
    ax1.set_title('{Patient_age}-{Patient_gender}'.format(**c_row))
    ax1.axis('off')
    
    c_segs = create_segmentation(c_img, c_row).astype(int)
    ax2.imshow(mark_boundaries(image=apply_softwindow(c_img), 
                               label_img=c_segs,
                               color=(0,1,0),
                              mode='thick'))
    ax2.set_title('Segmentation Map')
    
fig.savefig('over_withsegs.png', dpi = 300)
img_list = []
seg_list = []
path_list = []
from tqdm import tqdm_notebook
for (_, c_row) in tqdm_notebook(patient_df.iterrows()):
    c_img = read_hu(c_row['kaggle_path'])
    img_list+=[c_img]
    seg_list+=[create_segmentation(c_img, c_row).astype(bool)]
    path_list+=[c_row['File_name']]
from skimage.transform import resize
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
from skimage.util.montage import montage2d as montage
all_lesions = smart_stack([autocrop((img+1000)*seg,0)-1000 for img, seg in zip(img_list, seg_list)])
fig, ax1 = plt.subplots(1, 1, figsize = (15, 15))
ax1.imshow(montage(all_lesions), cmap = 'bone', vmin = -500, vmax = 400)
fig.savefig('montage.png', dpi = 300)
import h5py
with h5py.File('deeplesion.h5', 'w') as h:
    h.create_dataset('image', data=np.expand_dims(smart_stack(img_list, 0), -1), 
                     compression = 5)    
    h.create_dataset('mask', data=np.expand_dims(smart_stack(seg_list, 0), -1).astype(bool), 
                     compression = 5)    
    h.create_dataset('file_name', data=[x.encode('ascii') for x in path_list], 
                     compression = 0)    
# check the file
!ls -lh *.h5
with h5py.File('deeplesion.h5', 'r') as h:
    for k in h.keys():
        print(k, h[k].shape, h[k].dtype, h[k].size/1024**2)
