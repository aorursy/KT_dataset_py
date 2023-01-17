%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
base_tile_dir = os.path.join('..', 'input', 'kather_texture_2016_image_tiles_5000')
tile_df = pd.DataFrame({
    'path': glob(os.path.join(base_tile_dir, '*', '*', '*.tif'))
})
tile_df['file_id'] = tile_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
tile_df['cell_type'] = tile_df['path'].map(lambda x: os.path.basename(os.path.dirname(x))) 
tile_df['cell_type_idx'] = tile_df['cell_type'].map(lambda x: int(x.split('_')[0]))
tile_df['cell_type'] = tile_df['cell_type'].map(lambda x: x.split('_')[1])
tile_df['full_image_name'] = tile_df['file_id'].map(lambda x: x.split('_Row')[0])
tile_df['full_image_row'] = tile_df['file_id'].map(lambda x: int(x.split('_')[-3]))
tile_df['full_image_col'] = tile_df['file_id'].map(lambda x: int(x.split('_')[-1]))
tile_df.sample(3)
tile_df.describe(exclude=[np.number])
fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
# spatial distribution of tiles from the output image
fig, m_axs = plt.subplots(3, 3, figsize=(15, 15))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for c_ax, (c_type, c_rows) in zip(m_axs.flatten(), 
                                  tile_df.groupby('cell_type')):
    c_ax.scatter(c_rows['full_image_col'], c_rows['full_image_row'])
    c_ax.set_title(c_type)
    c_ax.set_xlim(tile_df['full_image_col'].min(), tile_df['full_image_col'].max())
    c_ax.set_ylim(tile_df['full_image_row'].min(), tile_df['full_image_row'].max())
# load in all of the images
from skimage.io import imread
tile_df['image'] = tile_df['path'].map(imread)
n_samples = 5
fig, m_axs = plt.subplots(8, n_samples, figsize = (4*n_samples, 3*8))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         tile_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)
rgb_info_df = tile_df.apply(lambda x: pd.Series({'{}_mean'.format(k): v for k, v in 
                                  zip(['Red', 'Green', 'Blue'], 
                                      np.mean(x['image'], (0, 1)))}),1)
gray_col_vec = rgb_info_df.apply(lambda x: np.mean(x), 1)
for c_col in rgb_info_df.columns:
    rgb_info_df[c_col] = rgb_info_df[c_col]/gray_col_vec
rgb_info_df['Gray_mean'] = gray_col_vec
rgb_info_df.sample(3)
tile_info_df = pd.concat([tile_df, rgb_info_df], 1, sort=True)
tile_info_df.sample(1)
sns.pairplot(tile_info_df[['Red_mean', 'Green_mean', 'Blue_mean', 'Gray_mean', 'cell_type']], 
             hue='cell_type', plot_kws = {'alpha': 0.5})
n_samples = 5
for sample_col in ['Red_mean', 'Green_mean', 'Blue_mean', 'Gray_mean']:
    fig, m_axs = plt.subplots(8, n_samples, figsize = (4*n_samples, 3*8))
    def take_n_space(in_rows, val_col, n):
        s_rows = in_rows.sort_values([val_col])
        s_idx = np.linspace(0, s_rows.shape[0]-1, n, dtype=int)
        return s_rows.iloc[s_idx]
    for n_axs, (type_name, type_rows) in zip(m_axs, 
                                             tile_info_df.sort_values(['cell_type']).groupby('cell_type')):

        for c_ax, (_, c_row) in zip(n_axs, 
                                    take_n_space(type_rows, 
                                                 sample_col,
                                                 n_samples).iterrows()):
            c_ax.imshow(c_row['image'])
            c_ax.axis('off')
            c_ax.set_title('{:2.2f}'.format(c_row[sample_col]))
        n_axs[0].set_title(type_name)
    fig.savefig('{}_samples.png'.format(sample_col), dpi=300)
from skimage.util import montage
rgb_stack = np.stack(tile_info_df.sort_values(['cell_type', 'Red_mean'])['image'].values, 0)
rgb_montage = np.stack([montage(rgb_stack[:, :, :, i]) for i in range(rgb_stack.shape[3])], -1)
plt.imshow(rgb_montage)
from skimage.io import imsave
imsave('full_dataset_montage.png', rgb_montage)
tile_info_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()
from PIL import Image
def package_mnist_df(in_rows, 
                     image_col_name = 'image',
                     label_col_name = 'cell_type_idx',
                     image_shape=(28, 28), 
                     image_mode='RGB',
                     label_first=False
                    ):
    out_vec_list = in_rows[image_col_name].map(lambda x: 
                                               np.array(Image.\
                                                        fromarray(x).\
                                                        resize(image_shape, resample=Image.LANCZOS).\
                                                        convert(image_mode)).ravel())
    out_vec = np.stack(out_vec_list, 0)
    out_df = pd.DataFrame(out_vec)
    n_col_names =  ['pixel{:04d}'.format(i) for i in range(out_vec.shape[1])]
    out_df.columns = n_col_names
    out_df['label'] = in_rows[label_col_name].values.copy()
    if label_first:
        return out_df[['label']+n_col_names]
    else:
        return out_df
from itertools import product
for img_side_dim, img_mode in product([8, 28, 64, 128], ['L', 'RGB']):
    if (img_side_dim==128) and (img_mode=='RGB'):
        # 128x128xRGB is a biggie
        break
    out_df = package_mnist_df(tile_info_df, 
                              image_shape=(img_side_dim, img_side_dim),
                             image_mode=img_mode)
    out_path = f'hmnist_{img_side_dim}_{img_side_dim}_{img_mode}.csv'
    out_df.to_csv(out_path, index=False)
    print(f'Saved {out_df.shape} -> {out_path}: {os.stat(out_path).st_size/1024:2.1f}kb')
