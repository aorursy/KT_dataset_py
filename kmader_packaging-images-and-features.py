%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams["figure.figsize"] = (15, 10)

plt.rcParams["figure.dpi"] = 300

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'gray' # grayscale looks better

from itertools import cycle

prop_cycle = plt.rcParams['axes.prop_cycle']

colors = prop_cycle.by_key()['color']
from pathlib import Path

import numpy as np

import pandas as pd

import os

from skimage.io import imread as imread

from skimage.util import montage

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

from skimage.color import label2rgb
base_dir = Path('../input/cartoon-set')

cartoon_df = pd.DataFrame({'path': list(base_dir.glob('*/*.*'))})

cartoon_df['dataset'] = cartoon_df['path'].map(lambda x: x.parent.stem)

cartoon_df['image_id'] = cartoon_df['path'].map(lambda x: x.stem)

cartoon_df['suffix'] = cartoon_df['path'].map(lambda x: x.suffix[1:])

flat_cartoon_df = cartoon_df.pivot_table(index=['dataset', 'image_id'], columns='suffix', values='path', aggfunc='first').reset_index().dropna()

flat_cartoon_df.sample(3)
def parse_csv(in_url):

    return {c_row[0]: c_row[1] # we don't care about total variants

            for c_row in pd.read_csv(in_url, header=None).to_dict('records')}

parse_csv(flat_cartoon_df.iloc[1]['csv'])
flat_cartoon_df['char_dict'] = flat_cartoon_df['csv'].map(parse_csv)

flat_cartoon_df.sample(3)
char_name_list = list(parse_csv(flat_cartoon_df.iloc[0]['csv']).keys())

for k in char_name_list:

    flat_cartoon_df[k] = flat_cartoon_df['char_dict'].map(lambda x: x[k])

flat_cartoon_df.drop(['csv', 'char_dict', 'image_id'], axis=1).to_csv('clean_char_features.csv', index=False)

flat_cartoon_df.sample(3)
sns.pairplot(flat_cartoon_df[char_name_list])
c_img = imread(flat_cartoon_df.iloc[1]['png'])

plt.imshow(c_img)
lin_steps = np.linspace(0, flat_cartoon_df.shape[0]-1, 6).astype(int)

fig, m_axs = plt.subplots(len(char_name_list), len(lin_steps), figsize=(12, 35))

for c_axs, c_name in zip(m_axs, char_name_list):

    c_paths = flat_cartoon_df.sort_values(c_name)['png'].values[lin_steps]

    c_axs[0].set_title(c_name)

    for c_ax, c_path in zip(c_axs, c_paths):

        c_ax.imshow(imread(c_path))

        c_ax.axis('off')
from PIL import Image

from tqdm import tqdm_notebook

import h5py

OUT_SIZE = (128, 128)

with h5py.File('out_cartoons.h5', 'w') as h:

    c_df = flat_cartoon_df.sort_values('image_id').reset_index(drop=True)

    for c_name in char_name_list:

        h.create_dataset(c_name, data=c_df[c_name].values, dtype='int16', compression='gzip')

    image_ds = h.create_dataset('image', shape=(c_df.shape[0],)+OUT_SIZE+(3,), dtype='uint8')

    for i, (_, c_row) in enumerate(tqdm_notebook(list(c_df.iterrows()))):

        image_ds[i] = np.array(Image.open(c_row['png']).resize(OUT_SIZE).convert('RGB'))

with h5py.File('out_cartoons.h5', 'r') as h:

    for k in h.keys():

        print(k, h[k].shape, h[k].dtype)