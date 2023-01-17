import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from skimage.util.montage import montage2d

from skimage.color import label2rgb

import os

import h5py
study_df = pd.read_csv(os.path.join('..', 'input', 'study_list.csv'))

study_df.sample(3) # show 3 random patients
%matplotlib inline

with h5py.File(os.path.join('..', 'input', 'lab_petct_vox_5.00mm.h5'), 'r') as p_data:

    fig, sb_mat = plt.subplots(3, 3, figsize=(10, 10), dpi = 250)

    (ax1s, ax2s, ax3s) = sb_mat.T

    for c_ax1, c_ax2, c_ax3, (p_id, ct_img), pt_img, lab_img in zip(ax1s, ax2s, ax3s,

                                   p_data['ct_data'].items(),

                                   p_data['pet_data'].values(),

                                   p_data['label_data'].values()

                                                           ):

        c_df = study_df[study_df['Patient ID']==p_id].head(1)

        c_dict = list(c_df.head(1).T.to_dict().values())[0]

        c_ax1.imshow(np.sum(ct_img,1).squeeze()[::-1,:], cmap = 'bone')

        c_ax1.set_title('CT:{Patient ID}\n{Site of primary STS} - {Grade}'.format(**c_dict))

        c_ax1.axis('off')

        

        c_ax2.imshow(np.sqrt(np.max(pt_img,1).squeeze()[::-1,:]), cmap = 'magma')

        c_ax2.set_title('PET\n(sqrt)'.format(**c_dict))

        c_ax2.axis('off')

        

        c_ax3.imshow(np.max(lab_img,1).squeeze()[::-1,:], cmap = 'gist_earth')

        c_ax3.set_title('Label'.format(**c_dict))

        c_ax3.axis('off')
%matplotlib inline

with h5py.File(os.path.join('..', 'input', 'lab_petct_vox_5.00mm.h5'), 'r') as p_data:

    fig, m_axs = plt.subplots(1, 1, figsize=(8, 8), dpi = 250)

    for c_ax, (p_id, p_img), lab_img in zip([m_axs], p_data['ct_data'].items(), p_data['label_data'].values()):

        c_df = study_df[study_df['Patient ID']==p_id].head(1)

        c_dict = list(c_df.head(1).T.to_dict().values())[0]

        montage_ct = montage2d(np.array(p_img[40:-80:1]))

        montage_ct += 1024

        montage_ct /= 2048

        montage_label = montage2d(np.array(lab_img[40:-80:1]).astype(np.float32)).astype(np.uint8)

        c_ax.imshow(label2rgb(montage_label, montage_ct.clip(0,1), bg_label = 0))

        c_ax.set_title('{Patient ID}\n{Site of primary STS} - {Grade}'.format(**c_dict))

        c_ax.axis('off')