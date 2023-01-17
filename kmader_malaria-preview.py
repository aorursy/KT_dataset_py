%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from pathlib import Path

from skimage.io import imread

from skimage.color import label2rgb

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

malaria_dir = Path('../input/malaria/malaria/')
train_df = pd.read_json(malaria_dir / 'training.json')

train_df['path'] = train_df['image'].map(lambda x: malaria_dir / x['pathname'][1:])

train_df['image_available'] = train_df['path'].map(lambda x: x.exists())

print(train_df.shape[0], 'images')

train_df = train_df.query('image_available')

print(train_df.shape[0], 'images available')

train_df.sample(5)
object_df = pd.DataFrame([dict(image=c_row['path'], **c_item) for _, c_row in train_df.iterrows() for c_item in c_row['objects']])

cat_dict = {v:k for k,v in enumerate(object_df['category'].value_counts().index, 1)}

print(object_df['category'].value_counts())

object_df.sample(3)
test_row = train_df.iloc[0]

def show_image_boxes(in_row, ax=None):

    """Show a row with bounding boxes"""

    if ax is None:

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    c_img = imread(in_row['path'])

    lab_img = np.zeros((c_img.shape[0], c_img.shape[1]), dtype='int')

    for c_obj in in_row['objects']:

        min_val = c_obj['bounding_box']['minimum']

        max_val = c_obj['bounding_box']['maximum']

        lab_id = cat_dict.get(c_obj['category'], 20)

        lab_img[slice(min_val['r'], max_val['r']), slice(min_val['c'], max_val['c'])] = lab_id

    ax.imshow(label2rgb(label=lab_img, image=c_img, bg_label=0))

show_image_boxes(test_row)
fig, m_axs = plt.subplots(3, 3, figsize=(30, 30))

valid_img_list = object_df.groupby('category').apply(lambda x: x.sample(2)).reset_index(drop=True)['image'].values.tolist()

for c_ax, (_, c_row) in zip(m_axs.flatten(), train_df[train_df['path'].isin(valid_img_list)].iterrows()):

    show_image_boxes(c_row, ax=c_ax)
fig.savefig('overlay_figure.png')