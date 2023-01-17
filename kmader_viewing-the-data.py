import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from skimage.util.montage import montage2d

import os

import h5py
study_df = pd.read_csv(os.path.join('..', 'input', 'study_list.csv'))

study_df.sample(3) # show 3 random patients
%matplotlib inline

with h5py.File(os.path.join('..', 'input', 'patient_images_lowres.h5'), 'r') as p_data:

    fig, m_axs = plt.subplots(2, 2, figsize=(12, 8), dpi = 250)

    for c_ax, (p_id, p_img) in zip(m_axs.flatten(), p_data['ct_data'].items()):

        c_df = study_df[study_df['Patient ID']==p_id].head(1)

        c_dict = list(c_df.head(1).T.to_dict().values())[0]

        c_ax.imshow(np.max(p_img,1).squeeze()[::-1,:], cmap = 'bone')

        c_ax.set_title('{Patient ID}\n{Site of primary STS} - {Grade}'.format(**c_dict))

        c_ax.set_aspect(2.5)

        c_ax.axis('off')
%matplotlib inline

with h5py.File(os.path.join('..', 'input', 'patient_images_lowres.h5'), 'r') as p_data:

    fig, m_axs = plt.subplots(2, 2, figsize=(12, 8), dpi = 250)

    for c_ax, (p_id, p_img) in zip(m_axs.flatten(), p_data['ct_data'].items()):

        c_df = study_df[study_df['Patient ID']==p_id].head(1)

        c_dict = list(c_df.head(1).T.to_dict().values())[0]

        c_ax.imshow(np.sum(p_img,1).squeeze()[::-1,:], cmap = 'bone')

        c_ax.set_title('{Patient ID}\n{Site of primary STS} - {Grade}'.format(**c_dict))

        c_ax.set_aspect(2.5)

        c_ax.axis('off')
%matplotlib inline

with h5py.File(os.path.join('..', 'input', 'patient_images_lowres.h5'), 'r') as p_data:

    fig, m_axs = plt.subplots(1, 1, figsize=(6, 6), dpi = 250)

    for c_ax, (p_id, p_img) in zip([m_axs], p_data['ct_data'].items()):

        c_df = study_df[study_df['Patient ID']==p_id].head(1)

        c_dict = list(c_df.head(1).T.to_dict().values())[0]

        c_ax.imshow(montage2d(np.array(p_img)), cmap = 'bone', vmin = -1024, vmax = 1024)

        c_ax.set_title('{Patient ID}\n{Site of primary STS} - {Grade}'.format(**c_dict))

        c_ax.axis('off')