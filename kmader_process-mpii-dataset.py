%matplotlib inline
import os
from glob import glob
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from skimage.util.montage import montage2d
from skimage.io import imread
from scipy.io import loadmat # for loading mat files
from tqdm import tqdm_notebook
root_mpi_dir = os.path.join('..', 'input', 'mpiigaze')
data_dir = os.path.join(root_mpi_dir, 'Data')
annot_dir = os.path.join(root_mpi_dir, 'Annotation Subset') # annotations the important part of the data
img_dir = os.path.join(data_dir, 'Original')
def read_annot(in_path):
    r_dir = os.path.splitext(os.path.basename(in_path))[0]
    c_df = pd.read_table(in_path, header = None, sep = ' ')
    c_df.columns = ['path' if i<0 else ('x{}'.format(i//2) if i % 2 == 0 else 'y{}'.format(i//2)) for i, x in enumerate(c_df.columns, -1)]
    c_df['path'] = c_df['path'].map(lambda x: os.path.join(img_dir, r_dir, x))
    c_df['group'] = r_dir
    c_df['exists'] = c_df['path'].map(os.path.exists)
    return c_df
all_annot_df = pd.concat([read_annot(c_path) for c_path in glob(os.path.join(annot_dir, '*'))], ignore_index=True)
print(all_annot_df.shape[0], 'annotations')
print('Missing %2.2f%%' % (100-100*all_annot_df['exists'].mean()))
all_annot_df = all_annot_df[all_annot_df['exists']].drop('exists', 1)
all_annot_df.sample(3)
group_view = all_annot_df.groupby('group').apply(lambda x: x.sample(2)).reset_index(drop = True)
fig, m_axs = plt.subplots(2, 3, figsize = (30, 10))
for (_, c_row), c_ax in zip(group_view.iterrows(), m_axs.flatten()):
    c_img = imread(c_row['path'])
    c_ax.imshow(c_img)
    for i in range(7):
        c_ax.plot(c_row['x{}'.format(i)], c_row['y{}'.format(i)], 's', label = 'xy{}'.format(i))
    c_ax.legend()
    c_ax.set_title('{group}'.format(**c_row))
from scipy.ndimage import zoom
def get_eyeball(in_row, eye_height = 30):
    c_img = imread(in_row['path'])
    min_x = int(in_row['x0'])
    max_x = int(in_row['x1'])
    
    mean_x = (min_x+max_x)/2
    wid_x = (max_x-min_x)
    zoom_factor = 55.0/wid_x
    
    mean_y = (in_row['y0']+in_row['y1'])/2
    eye_height = 1/zoom_factor*35
    # normalized pupil position
    pup_v = 2*zoom_factor*(in_row['x6']-mean_x)/wid_x, 2*zoom_factor*(in_row['y6']-mean_y)/eye_height
    
    min_y = int(mean_y-eye_height//2)
    max_y = int(mean_y+eye_height//2)
    out_img = c_img[min_y:max_y, min_x:max_x]
    rs_img = zoom(out_img, (zoom_factor, zoom_factor, 1))
    return rs_img, pup_v
fig, m_axs = plt.subplots(2, 3, figsize = (30, 10))
for (_, c_row), c_ax in zip(group_view.iterrows(), m_axs.flatten()):
    c_img, c_vec = get_eyeball(c_row)
    c_ax.imshow(c_img)
    c_ax.quiver([55//2], [35//2], 20*c_vec[1], 20*c_vec[0], units = 'xy', scale = 1, color = 'red')
    c_ax.set_title('({0:2.2f}, {1:2.2f}) - {group}'.format(*c_vec, **c_row))
all_annot_df['eyeball'] = all_annot_df.apply(get_eyeball, 1)
all_annot_df['pupil_x'] = all_annot_df['eyeball'].map(lambda x: x[1][0])
all_annot_df['pupil_y'] = all_annot_df['eyeball'].map(lambda x: x[1][1])
all_annot_df['eyeball'] = all_annot_df['eyeball'].map(lambda x: x[0])
all_annot_df.sample(3)
fig, m_axs = plt.subplots(2, 4, figsize = (20, 20))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
from itertools import product
for (ax_dist, ax_min, ax_mean, ax_max), n_ax in zip(m_axs, ['pupil_x', 'pupil_y']):
    # use random sampling to get a better feeling
    c_vec = all_annot_df[n_ax]
    ax_dist.hist(c_vec.values, 30)
    ax_dist.axis('on')
    j = c_vec.idxmin()
    ax_min.imshow(all_annot_df.loc[j]['eyeball'])
    ax_min.set_title('min {}: {:2.2f}'.format(n_ax, all_annot_df.loc[j][n_ax]))
    
    k = c_vec.idxmax()
    ax_max.imshow(all_annot_df.loc[k]['eyeball'])
    ax_max.set_title('max {}: {:2.2f}'.format(n_ax, all_annot_df.loc[k][n_ax]))
    
    p = np.abs(c_vec-np.mean(c_vec)).idxmin()
    ax_mean.imshow(all_annot_df.loc[p]['eyeball'])
    ax_mean.set_title('mean: {}: {:2.2f}'.format(n_ax, all_annot_df.loc[p][n_ax]))
def parse_mat(in_path):
    in_dat = loadmat(in_path, squeeze_me = True, struct_as_record = True)
    vec1_load, img_load,vec2_load = in_dat['data'].tolist()[1].tolist()
    return vec1_load, img_load, vec2_load
def mat_to_df(in_path):
    vec1_load, img_load, vec2_load = parse_mat(in_path)
    c_df = pd.DataFrame(dict(img=[x for x in img_load], 
                             vec1=[x for x in vec1_load],
                            vec2=[x for x in vec2_load]))
    c_df['group'] = os.path.basename(os.path.dirname(in_path))
    c_df['day'] = os.path.splitext(os.path.basename(in_path))[0]
    return c_df
def safe_mat_to_df(in_path):
    try:
        return mat_to_df(in_path)
    except ValueError as e:
        print('ValueError', e, in_path)
        return None
mat_files = glob(os.path.join(root_mpi_dir, '..', 'normalized', '*', '*.mat'))
print(len(mat_files), 'normalized files found')
all_norm_df = pd.concat([safe_mat_to_df(in_path) for in_path in tqdm_notebook(mat_files)], ignore_index=True)
all_norm_df.sample(3)
print(all_norm_df.shape[0], 'images loaded')
group_view = all_norm_df.groupby('group').apply(lambda x: x.sample(1)).reset_index(drop = True)
fig, m_axs = plt.subplots(3, 3, figsize = (20, 20))
for (_, c_row), c_ax in zip(group_view.iterrows(), m_axs.flatten()):
    c_ax.imshow(c_row['img'], cmap = 'gray')
    c_ax.legend()
    c_ax.set_title('{group}'.format(**c_row))
for v in ['vec1', 'vec2']:
    for i, x_dim in enumerate('xyz'):
        all_norm_df['{}_{}'.format(v, x_dim)] = all_norm_df[v].map(lambda x: x[i])
all_norm_df.sample(3)
fig, m_axs = plt.subplots(6, 4, figsize = (20, 20))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
from itertools import product
for (ax_dist, ax_min, ax_mean, ax_max), (v, (i, x)) in zip(m_axs, product(['vec1', 'vec2'], enumerate('xyz'))):
    # use random sampling to get a better feeling
    c_vec = all_norm_df.sample(1000)['{}_{}'.format(v, x)]
    ax_dist.hist(c_vec.values, 30)
    ax_dist.axis('on')
    j = c_vec.idxmin()
    ax_min.imshow(all_norm_df.iloc[j]['img'], cmap = 'bone')
    ax_min.set_title('min {}_{}: {:2.2f}'.format(v, x, all_norm_df.iloc[j]['{}_{}'.format(v, x)]))
    
    k = c_vec.idxmax()
    ax_max.imshow(all_norm_df.iloc[k]['img'], cmap = 'bone')
    ax_max.set_title('max {}_{}: {:2.2f}'.format(v, x, all_norm_df.iloc[k]['{}_{}'.format(v, x)]))
    
    p = np.abs(c_vec-np.mean(c_vec)).idxmin()
    ax_mean.imshow(all_norm_df.iloc[p]['img'], cmap = 'bone')
    ax_mean.set_title('mean: {}_{}: {:2.2f}'.format(v, x, all_norm_df.iloc[p]['{}_{}'.format(v, x)]))

