import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from cv2 import imread, createCLAHE # read and equalize images

from glob import glob

%matplotlib inline

import matplotlib.pyplot as plt
all_xray_df = pd.read_csv('../input/Data_Entry_2017.csv')

all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input', 'images*', '*', '*.png'))}

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x[:-1]))

all_xray_df.sample(3)
label_counts = all_xray_df['Finding Labels'].value_counts()[:15]

fig, ax1 = plt.subplots(1,1,figsize = (12, 8))

ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

ax1.set_xticks(np.arange(len(label_counts))+0.5)

_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

from itertools import chain

all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))

print('All Labels', all_labels)

for c_label in all_labels:

    if len(c_label)>1: # leave out empty labels

        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

all_xray_df.sample(3)
# since we can't have everything make a nice subset

# weight is 0.1 + number of findings

sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 1e-1

sample_weights /= sample_weights.sum()

all_xray_df = all_xray_df.sample(18000, weights=sample_weights)



label_counts = all_xray_df['Finding Labels'].value_counts()[:15]

fig, ax1 = plt.subplots(1,1,figsize = (12, 8))

ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

ax1.set_xticks(np.arange(len(label_counts))+0.5)

_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
import h5py

from tqdm import tqdm



def write_df_as_hdf(out_path,

                    out_df,

                    compression='gzip'):

    with h5py.File(out_path, 'w') as h:

        for k, arr_dict in tqdm(out_df.to_dict().items()):

            try:

                s_data = np.stack(arr_dict.values(), 0)

                try:

                    h.create_dataset(k, data=s_data, compression=

                    compression)

                except TypeError as e:

                    try:

                        h.create_dataset(k, data=s_data.astype(np.string_),

                                         compression=compression)

                    except TypeError as e2:

                        print('%s could not be added to hdf5, %s' % (

                            k, repr(e), repr(e2)))

            except ValueError as e:

                print('%s could not be created, %s' % (k, repr(e)))

                all_shape = [np.shape(x) for x in arr_dict.values()]

                warn('Input shapes: {}'.format(all_shape))
write_df_as_hdf('chest_xray.h5', all_xray_df)
# show what is inside

with h5py.File('chest_xray.h5', 'r') as h5_data:

    for c_key in h5_data.keys():

        print(c_key, h5_data[c_key].shape, h5_data[c_key].dtype)
from skimage.transform import resize

OUT_DIM = (128, 128)

clahe_tool = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def imread_and_normalize(im_path):

    img_data = np.mean(imread(im_path), 2).astype(np.uint8)

    n_img = (255*resize(img_data, OUT_DIM, mode = 'constant')).clip(0,255).astype(np.uint8)

    return np.expand_dims(n_img, -1)





test_img = imread_and_normalize(all_xray_df['path'].values[0])

plt.matshow(test_img[:,:,0])
%%time

# time for 100 images

pre_read_tiles = np.stack(all_xray_df.sample(100)['path'].map(imread_and_normalize).values,0)
# might as well show-em if we have em

from skimage.util.montage import montage2d

fig, ax1 = plt.subplots(1,1, figsize = (12,12))

ax1.imshow(montage2d(pre_read_tiles[:,:,:,0]), cmap = 'bone')

fig.savefig('overview.png', dpi = 300)
# preallocate output

out_image_arr = np.zeros((all_xray_df.shape[0],)+OUT_DIM+(1,), dtype=np.uint8)

if False:

    # a difficult to compress array for size approximations

    out_image_arr = np.random.uniform(0, 255,

                                  size = (all_xray_df.shape[0],)+OUT_DIM+(1,)).astype(np.uint8)
for i, c_path in enumerate(tqdm(all_xray_df['path'].values)):

    out_image_arr[i] = imread_and_normalize(c_path)
# create a new h5 with everything we want in it (kaggle fs does weird caching)

with h5py.File('chest_xray.h5', 'a') as h5_data:

    h5_data.create_dataset('images', data = out_image_arr, compression = None)

with h5py.File('chest_xray.h5', 'r') as h5_data:

    for c_key in h5_data.keys():

        print(c_key, h5_data[c_key].shape, h5_data[c_key].dtype)
print('Output File-size %2.2fMB' % (os.path.getsize('chest_xray.h5')/1e6))