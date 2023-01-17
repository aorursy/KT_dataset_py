%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from IPython.display import Image, display, SVG, clear_output

plt.rcParams["figure.figsize"] = (6, 6)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'gray' # grayscale looks better
import numpy as np

from skimage.io import imread

import pandas as pd

from scipy.io import loadmat

from pathlib import Path
import inspect

import doctest

import copy

import functools

def autotest(func):

    globs = copy.copy(globals())

    globs.update({func.__name__: func})

    doctest.run_docstring_examples(

        func, globs, verbose=True, name=func.__name__)

    return func

base_dir = Path('../input/mo2cap2/test_data/')
all_images_df = pd.DataFrame({'path': list(base_dir.glob('**/*.*'))})

all_images_df['suffix'] = all_images_df['path'].map(lambda x: x.suffix[1:].lower())

all_images_df = all_images_df[all_images_df['suffix'].isin(['jpg', 'png'])]

all_images_df['scene'] = all_images_df['path'].map(lambda x: x.parent.stem)

all_images_df['timestamp'] = pd.to_datetime(all_images_df['path'].map(lambda x: x.stem.split('_')[-1][:-5]), format='%Y-%m-%d-%H%M%S', errors='ignore')

offset_dict = {'weipeng_studio': 386, 'olek_outdoor': 157}

offset = all_images_df['scene'].map(lambda x: offset_dict.get(x,0))+1 # +1 for python instead of matlab indexing

all_images_df['idx'] = all_images_df['path'].map(lambda x: int(x.stem.split('_')[-1][-4:]))-offset

all_images_df.sample(3)
all_images_df.groupby('scene').agg({'idx': ['min', 'max', 'count']})
all_images_df['scene'].value_counts()
fig, m_axs = plt.subplots(8, 8, figsize=(30, 30))

for c_ax, (_, c_row) in zip(m_axs.flatten(), all_images_df.groupby('scene').apply(lambda x: x.sample(np.prod(m_axs.shape)//2)).iterrows()):

    c_ax.imshow(imread(c_row['path']))

    c_ax.set_title('{scene}-{idx}'.format(**c_row))

    c_ax.axis('off')
body_dict = { 1:'neck', 2:'Rsho', 3:'Relb', 4:'Rwri', 5:'Lsho', 6:'Lelb', 7:'Lwri', 8:'Rhip', 9:'Rkne', 10:'Rank', 11:'Rtoe', 12: 'Lhip', 13:'Lkne', 14:'Lank', 15:'Ltoe'}

mat_dict = {k: loadmat(k)['pose_gt'] for k in base_dir.glob('**/*.mat')}
fig = plt.figure(figsize=(20, 10))

for ax_id, (k,v) in enumerate(mat_dict.items()):

    ax = fig.add_subplot((121+ax_id),projection='3d')

    for i, bp_name in body_dict.items():

        ax.plot(v[:, i-1, 2], v[:, i-1, 0], v[:, i-1, 1], '.', label=bp_name)

    ax.legend()

    ax.set_title(k.stem)