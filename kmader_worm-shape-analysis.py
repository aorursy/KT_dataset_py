import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from skimage.io import imread

import matplotlib.pyplot as plt

import seaborn as sns
all_tif_images=glob('../input/BBBC010_v1_images/*_w1_*.tif')

all_fg_images=glob('../input/BBBC010_v1_foreground/*.png')

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

# clearly this is not the case

# <plate>_<wellrow>_<wellcolumn>_<wavelength>_<fileid>.tif

# Columns 1-12 are positive controls treated with ampicillin. Columns 13-24 are untreated negative controls.

# we apply a new rule

# 1649_1109_0003_Amp5-1_B_20070424_A01_w1_9E84F49F-1B25-4E7E-8040-D1BB2D7E73EA.tif

# junk_junk_junk_junk_junk_date_RowCol_wavelength_id.tif



image_df['plate_rc']=image_df['base_name'].map(lambda x: x.split('_')[6])

image_df['row']=image_df['plate_rc'].map(lambda x: x[0:1])

image_df['column']=image_df['plate_rc'].map(lambda x: int(x[1:]))

image_df['treated']=image_df['column'].map(lambda x: 'ampicillin' if x<13 else 'negative control')

image_df['wavelength']=image_df['base_name'].map(lambda x: x.split('_')[7])



image_df['mask_path']=image_df['plate_rc'].map(lambda x: '../input/BBBC010_v1_foreground/{}_binary.png'.format(x))

print('Loaded',image_df.shape[0],'datasets')

image_df.sample(3)
%%time

from skimage.measure import label

from skimage.measure import regionprops

image_df['worm_image']=image_df['mask_path'].map(lambda x: imread(x)[:,:,0]>0)

image_df['worm_labels']=image_df['worm_image'].map(lambda x: label(x))

image_df['worm_regions']=image_df['worm_labels'].map(lambda x: regionprops(x))

image_df['worm_volume_fraction']=image_df['worm_image'].map(lambda x: np.mean(x))

image_df['worm_nsegments']=image_df['worm_labels'].map(lambda x: np.max(x))

image_df['worm_p2a_ratio']=image_df['worm_regions'].map(lambda x: np.mean([rp.perimeter/rp.area for rp in x]))

image_df.sample(3)
sns.pairplot(image_df.drop(['column'],1),hue='treated')