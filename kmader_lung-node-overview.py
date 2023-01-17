from __future__ import print_function, division

import numpy as np

import csv

from glob import glob

import pandas as pd

import os

from tqdm import tqdm

output_path = os.path.join('..','input')

import matplotlib.pyplot as plt

from skimage.util.montage import montage2d

from skimage.color import label2rgb

%matplotlib inline
import h5py

with h5py.File(os.path.join(output_path, 'all_patches.hdf5'), 'r') as luna_h5:

    all_slices = luna_h5['ct_slices'].value

    all_classes = luna_h5['slice_class'].value

    print('data', all_slices.shape, 'classes', all_classes.shape)
from skimage.util.montage import montage2d

fig, (ax1, ax2) = plt.subplots(1,2,figsize = (12, 6))

plt_args = dict(cmap = 'bone', vmin = -600, vmax = 300)

ax1.imshow(montage2d(all_slices[np.random.choice(np.where(all_classes>0.5)[0],size = 64)]), **plt_args)

ax1.set_title('Malignant Tiles')

ax2.imshow(montage2d(all_slices[np.random.choice(np.where(all_classes<0.5)[0],size = 64)]), **plt_args)

ax2.set_title('Benign Tiles')
from skimage.filters import laplace

all_lesion_df = pd.DataFrame(dict(les_img = [x for x in all_slices], 

                                  malignant = all_classes[:,0]))

all_lesion_df['mean'] = all_lesion_df['les_img'].map(np.mean)

all_lesion_df['std'] = all_lesion_df['les_img'].map(np.std)

all_lesion_df['mean_lap'] = all_lesion_df['les_img'].map(lambda x: np.mean(laplace(x.clip(-600,0)/600)))

all_lesion_df['les_type'] = all_lesion_df['malignant'].map(lambda x: 'malignant' if x>0.5 else 'benign')

all_lesion_df.sample(3)
import seaborn as sns

sns.pairplot(all_lesion_df, hue = 'les_type')
sns.factorplot(x = 'les_type', y = 'mean', kind = 'box', data = all_lesion_df)
sns.factorplot(x = 'les_type', y = 'std', kind = 'box', data = all_lesion_df)
sns.factorplot(x = 'les_type', y = 'mean_lap', kind = 'box', data = all_lesion_df)