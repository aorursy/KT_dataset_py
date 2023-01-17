import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from skimage.io import imread

import matplotlib.pyplot as plt

import seaborn as sns
lung_df=pd.read_csv(os.path.join('..','input','lung_stats.csv'))

lung_df.sample(2)
%matplotlib inline

sns.pairplot(lung_df)
# add the image and mask paths

lung_df['image_path']=lung_df['img_id'].map(lambda x: os.path.join('..','input','2d_images',x))

lung_df['mask_path']=lung_df['img_id'].map(lambda x: os.path.join('..','input','2d_masks',x))
%matplotlib inline

for _, c_row in lung_df.sample(1).iterrows():

    c_img=imread(c_row['image_path'])

    c_mask=imread(c_row['mask_path'])>0

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))

    ax1.imshow(c_img)

    ax1.set_title('CT Image\nVF:{lung_volume_fraction}\nPD95:{lung_pd95_hu}'.format(**c_row))

    ax2.set_title('Manually Segmented Mask')

    ax2.imshow(c_mask)
import nibabel as nib

from glob import glob

all_nifti_imgs = glob(os.path.join('..', 'input', '3d_images', 'IMG_*.nii.gz'))

nifti_img = nib.load(all_nifti_imgs[0])
from skimage.util.montage import montage2d

fig, ax1 = plt.subplots(1,1,figsize = (9,9))

ax1.imshow(montage2d(nifti_img.get_data()[::2]), cmap = 'bone')

ax1.axis('off')
all_mask_imgs = glob(os.path.join('..', 'input', '3d_images', 'MASK_*.nii.gz'))

mask_img = nib.load(all_mask_imgs[0])
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

p = mask_img.get_data()[::-1, ::2, ::2].swapaxes(1,2)

cmap = plt.cm.get_cmap('nipy_spectral_r')

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')



verts, faces = measure.marching_cubes(p, 0)

mesh = Poly3DCollection(verts[faces], alpha=0.25, edgecolor='none', linewidth = 0.1)



mesh.set_edgecolor([1, 0, 0])

ax.add_collection3d(mesh)



ax.set_xlim(0, p.shape[0])

ax.set_ylim(0, p.shape[1])

ax.set_zlim(0, p.shape[2])



ax.view_init(45, 45)

fig.savefig('lung_3d.pdf')