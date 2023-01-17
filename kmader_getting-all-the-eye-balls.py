%matplotlib inline
import numpy as np
import pandas as pd
from glob import glob
import os
import matplotlib.pyplot as plt
from skimage.io import imread
helen_dir = '../input/'
def read_annot_file(in_path):
    with open(in_path, 'r') as f:
        file_id = f.readline().strip()
    out_df = pd.read_csv(in_path, skiprows=1, header=None)
    out_df.columns = ['x', 'y']
    out_df['file_id'] = file_id
    return out_df
annot_df = pd.concat([
    read_annot_file(c_path).reset_index()
    for c_path in 
    glob(os.path.join(helen_dir, 'annotation', '*', '*'))
])
annot_df.sample(3)
all_image_dict = {os.path.splitext(f)[0]: os.path.join(p, f) 
              for p, _, files in os.walk(helen_dir) 
              for f in files if f.upper().endswith('JPG')}
annot_df['path'] = annot_df['file_id'].map(all_image_dict.get)
annot_df.dropna(inplace=True)
fig, m_axs = plt.subplots(3, 3, figsize = (15, 15))
for c_ax, (c_path, c_rows) in zip(m_axs.flatten(), 
                                  annot_df.groupby('path')):
    img = imread(c_path)
    c_ax.imshow(img)
    c_ax.plot(c_rows['x'], c_rows['y'], 'r.')
fig, m_axs = plt.subplots(3, 3, figsize = (30, 30))
marker_id = ['face', 'nose', 'mouth_inner', 'mouth_outer', 'r_eye', 'l_eye', 'r_eyebrow', 'l_eyebrow']
marker_split = np.cumsum([0, 41, 17, 28, 28, 20, 20, 20, 20])
marker_dict = {marker_id[i]: (marker_split[i], marker_split[i+1]) for i in range(len(marker_split)-1)}
for c_ax, (c_path, c_rows) in zip(m_axs.flatten(), 
                                  annot_df.groupby('path')):
    img = imread(c_path)
    c_ax.imshow(img)
    for label, (start, end) in marker_dict.items():
        n_rows = c_rows.query(f'index>={start} and index<={end}')
        c_ax.plot(n_rows['x'], n_rows['y'], '.', label=label)
    c_ax.legend()
point_map = {i: [k for k, (n,x) in marker_dict.items() if n<=i<x] 
 for i in range(annot_df['index'].max()+1) }
annot_df['body_part'] = annot_df['index'].map(lambda x: point_map.get(x)[0])
annot_df['body_part'].value_counts()
x_pad, y_pad = 35, 25
fig, m_axs = plt.subplots(2, 15, figsize = (30, 4))
filt_df = annot_df[annot_df['body_part'].isin(['l_eye', 'r_eye'])]
for c_axs, ((c_path, c_id), c_rows) in zip(m_axs.T, 
                                  filt_df.groupby(['path', 'file_id'])):
    img = imread(c_path)
    for c_ax, (body_part, n_rows) in zip(c_axs, c_rows.groupby('body_part')):
        x_min = int(n_rows['x'].min()-x_pad)
        x_max = int(n_rows['x'].max()+x_pad)
        y_min = int(n_rows['y'].min()-y_pad)
        y_max = int(n_rows['y'].max()+y_pad)
        roi = img[y_min:y_max, x_min:x_max]
        
        c_ax.imshow(roi)
        c_ax.set_title(f'{body_part}-{roi.shape[:2]}')
        c_ax.axis('off')
from tqdm import tqdm_notebook
all_rois = []
for ((c_path, c_id), c_rows) in tqdm_notebook(filt_df.groupby(['path', 'file_id'])):
    img = imread(c_path)
    for (body_part, n_rows) in c_rows.sort_values('body_part').groupby('body_part'):
        x_min = int(n_rows['x'].min()-x_pad)
        x_max = int(n_rows['x'].max()+x_pad)
        y_min = int(n_rows['y'].min()-y_pad)
        y_max = int(n_rows['y'].max()+y_pad)
        all_rois += [img[y_min:y_max, x_min:x_max]]
pd.DataFrame([{'size': c_roi.shape[0]/c_roi.shape[1]} for c_roi in all_rois])['size'].hist()
pd.DataFrame([{'x': c_roi.shape[0], 'y': c_roi.shape[1]} for c_roi in all_rois]).describe()
from PIL import Image
out_shape = (55, 35)
fig, m_axs = plt.subplots(5, 5, figsize = (10, 10))
for c_ax, c_roi in zip(m_axs.flatten(), all_rois):
    k = Image.fromarray(c_roi)
    c_ax.imshow(k.resize(out_shape, resample=Image.BICUBIC))
import h5py
with h5py.File('eye_balls_rgb.h5', 'w') as f:
    out_image = f.create_dataset('image', 
                                 shape=(len(all_rois), out_shape[1], out_shape[0], 3), 
                                chunks=(1, out_shape[1], out_shape[0], 3), 
                                 dtype=np.uint8,
                                compression=8)
    for i, c_roi in enumerate(tqdm_notebook(all_rois)):
        out_image[i, :, :, :] = Image.fromarray(c_roi).resize(out_shape, resample=Image.BICUBIC)
!ls -lh *.h5
with h5py.File('eye_balls_rgb.h5', 'r') as f:
    print(f['image'].shape)
    print(f['image'].value.mean(), f['image'].value.std())
