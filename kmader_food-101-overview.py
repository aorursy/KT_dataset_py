import h5py

import os

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
sample_path = os.path.join('..', 'input', 'food_c101_n10099_r64x64x3.h5')

with h5py.File(sample_path, 'r') as n_file:

    print('Data Size:', n_file['images'].shape)

    im_label = n_file['category'].value

    label_names = [x.decode() for x in n_file['category_names'].value]

    fig, (ax1) = plt.subplots(1, 1, figsize = (4, 14))

    v_sum = np.sum(im_label, 0)

    x_coord = np.arange(im_label.shape[1])

    ax1.barh(x_coord, v_sum)

    out_ticks = [(c_x, c_label) for c_sum, c_x, c_label in zip(v_sum, x_coord, label_names) if c_sum>0]

    ax1.set_yticks([c_x for c_x, c_label in out_ticks])

    _ = ax1.set_yticklabels([c_label for c_x, c_label in out_ticks], rotation=0, fontsize = 8)

    ax1.set_ylim(0, x_coord.max())
sample_imgs = 25

with h5py.File(sample_path, 'r') as n_file:

    total_imgs = n_file['images'].shape[0]

    read_idxs = slice(0,sample_imgs)

    im_data = n_file['images'][read_idxs]

    im_label = n_file['category'].value[read_idxs]

    label_names = [x.decode() for x in n_file['category_names'].value]
fig, m_ax = plt.subplots(5, 5, figsize = (12, 12))

for c_ax, c_label, c_img in zip(m_ax.flatten(), im_label, im_data):

    c_ax.imshow(c_img if c_img.shape[2]==3 else c_img[:,:,0], cmap = 'gray')

    c_ax.axis('off')

    c_ax.set_title(label_names[np.argmax(c_label)])
from glob import glob

for c_path in glob(os.path.join('..', 'input', 'food_*.h5')):

    with h5py.File(c_path, 'r') as n_file:

        total_imgs = n_file['images'].shape[0]

        read_idxs = slice(0,sample_imgs)

        im_data = n_file['images'][read_idxs]

        im_label = n_file['category'].value[read_idxs]

        label_names = [x.decode() for x in n_file['category_names'].value]

    fig, m_ax = plt.subplots(5, 5, figsize = (12, 12))

    for c_ax, c_label, c_img in zip(m_ax.flatten(), im_label, im_data):

        c_ax.imshow(c_img if c_img.shape[2]==3 else c_img[:,:,0], cmap = 'gray')

        c_ax.axis('off')

        c_ax.set_title(label_names[np.argmax(c_label)])

    fig.savefig('overview_{}.png'.format(os.path.basename(c_path)))