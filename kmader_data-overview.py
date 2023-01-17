import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import matplotlib.pyplot as plt # making plots

from skimage.util.montage import montage2d # showing a montage

import os

import seaborn as sns

base_path = '../input/MultiSpectralImages'
all_files = glob(os.path.join(base_path, '*'))

label_df = pd.read_csv(os.path.join(base_path, 'Labels.csv'))

label_df = label_df[['Label', 'FileName']] # other columns are NaNs

label_df['Number'] = label_df['Label'].map(lambda x: x.split(' ')[-1])

label_df['Color'] = label_df['Label'].map(lambda x: x.split(' ')[0])

print('Number of numbers',label_df.shape)

label_df.sample(3)
test_row = list(label_df.sample(1).T.to_dict().values())[0]

test_image = pd.read_csv(os.path.join(base_path, test_row['FileName']))

test_image.sample(3)
# examine the interesting bits of signal (not pegged at 255)

sns.pairplot(test_image.query('Channel0<255'))
x_min, x_max = test_image['X'].min(), test_image['X'].max()

y_min, y_max = test_image['Y'].min(), test_image['Y'].max()

out_dims = (y_max-y_min+1, x_max-x_min+1)
assert np.max(np.diff(test_image['X'].values.reshape(out_dims),1)) == 1, "Ensure the X dimensions can be reshaped"

assert np.max(np.diff(test_image['Y'].values.reshape(out_dims),1,0)) == 1, "Ensure the Y dimensions can be reshaped"
test_hs_image = np.stack([test_image['Channel{}'.format(i)].values.reshape(out_dims, order = 'C') for i in range(10)],0)

print(test_hs_image.shape)
fig, (m_axs) = plt.subplots(3,3, figsize = (13,13))

for i, c_ax in enumerate(m_axs.flatten()):

    c_ax.matshow(test_hs_image[i], cmap = 'gray')

    c_ax.set_title('{Color} {Number}\n(Channel {channel})'.format(channel = i, **test_row))

    c_ax.axis('off')
def read_image(in_csv_name):

    try:

        cur_img = pd.read_csv(os.path.join(base_path, in_csv_name))

    except:

        return None

    x_min, x_max = cur_img['X'].min(), cur_img['X'].max()

    y_min, y_max = cur_img['Y'].min(), cur_img['Y'].max()

    out_dims = (y_max-y_min+1, x_max-x_min+1)

    assert np.max(np.diff(cur_img['X'].values.reshape(out_dims),1)) == 1, "Ensure the X dimensions can be reshaped"

    assert np.max(np.diff(cur_img['Y'].values.reshape(out_dims),1,0)) == 1, "Ensure the Y dimensions can be reshaped"

    return np.stack([cur_img['Channel{}'.format(i)].values.reshape(out_dims, order = 'C') for i in range(10)],0)
%%time

# we can't process all of the data since it takes too long

subset_label_df = label_df.sample(50)

# read all of the images

subset_label_df['Image'] = subset_label_df['FileName'].map(read_image)
fig, m_axs = plt.subplots(10, 1, figsize = (20, 10))

for (i, n_row), c_ax in zip(subset_label_df.iterrows(), m_axs):

    c_ax.imshow(montage2d(n_row['Image'][:,100:250, 100:250], # crop the middle

                          grid_shape = (1, n_row['Image'].shape[0])), 

                cmap = 'gray')

    c_ax.axis('off')

    c_ax.set_title('{Color} {Number}'.format(**n_row))
np.savez_compressed('all_images.npz', 

                    images = subset_label_df['Image'], 

                    color = subset_label_df['Color'],

                   number = subset_label_df['Number'])