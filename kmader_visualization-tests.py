import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from skimage.io import imread

import matplotlib.pyplot as plt
all_tif_images=glob('../input/BBBC010_v1_images/*_w1_*.tif')



image_df=pd.DataFrame([{'gfp_path': f} for f in all_tif_images])

def _get_light_path(in_path):

    w2_path='_w2_'.join(in_path.split('_w1_'))

    glob_str='_'.join(w2_path.split('_')[:-1]+['*.tif'])

    m_files=glob(glob_str)

    if len(m_files)>0:

        return m_files[0]

    else:

        return None

image_df['light_path']=image_df['gfp_path'].map(_get_light_path)

image_df=image_df.dropna()

image_df['base_name']=image_df['gfp_path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])

image_df['row_col']=image_df['base_name'].map(lambda x: x.split('_')[6])

image_df['row']=image_df['row_col'].map(lambda x: x[0:1])

image_df['column']=image_df['row_col'].map(lambda x: int(x[1:]))

image_df['treated']=image_df['column'].map(lambda x: x<13)



image_df['mask_path']=image_df['row_col'].map(lambda x: '../input/BBBC010_v1_foreground/{}_binary.png'.format(x))

print('Loaded',image_df.shape[0],'datasets')

image_df.sample(3)
%matplotlib inline

test_image_row=list(image_df.query('treated').sample(1).T.to_dict().values())[0]

test_img=imread(test_image_row['light_path'])

test_gfp=imread(test_image_row['gfp_path'])

test_bg=imread(test_image_row['mask_path'])[:,:,0]>0

print('Test Image:',test_img.shape)

fig, (ax_light,ax_gfp, ax2 ,ax3) = plt.subplots(1,4, figsize = (15,3))

ax_light.imshow(test_img,cmap='gray')

ax_light.set_title('Light-field Image'.format(**test_image_row))



ax_gfp.imshow(test_gfp,cmap='gray')

ax_gfp.set_title('GFP Image'.format(**test_image_row))



ax2.hist(test_img.ravel())

ax2.set_title('Image Intensity Distribution')

ax3.imshow(test_bg, cmap = 'bone')

ax3.set_title('Segmented')
norm_cimg=lambda cmap,x: cmap((x-x.mean())/(1*x.std())+0.5)[:,:,:3]

fig,ax1=plt.subplots(1,1,figsize=(10,10),dpi=300)

ax1.imshow(0.5*norm_cimg(plt.cm.bone,test_img)+

           0.25*norm_cimg(plt.cm.Greens,test_gfp)+

          0.25*norm_cimg(plt.cm.magma,test_bg),interpolation='lanczos')

ax1.axis('equal')

ax1.axis('off')

fig.savefig('cover_img.jpg')