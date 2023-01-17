# extract the files

!tar -xf ../input/all-mias.tar.gz
import os

import matplotlib.pyplot as plt

import numpy as np

import re

from glob import glob

import pandas as pd

def read_pgm(filename, byteorder='>'):

    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """

    with open(filename, 'rb') as f:

        buffer = f.read()

    try:

        header, width, height, maxval = re.search(

            b"(^P5\s(?:\s*#.*[\r\n])*"

            b"(\d+)\s(?:\s*#.*[\r\n])*"

            b"(\d+)\s(?:\s*#.*[\r\n])*"

            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()

    except AttributeError:

        raise ValueError("Not a raw PGM file: '%s'" % filename)

    return np.frombuffer(buffer,

                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',

                            count=int(width)*int(height),

                            offset=len(header)

                            ).reshape((int(height), int(width)))
all_cases_df = pd.read_table('../input/Info.txt', delimiter=' ')

all_cases_df = all_cases_df[all_cases_df.columns[:-1]] # drop last column

all_cases_df['path'] = all_cases_df['REFNUM'].map(lambda x: '%s.pgm' % x)

all_cases_df.sample(3)
# load all the scans

all_cases_df['scan'] = all_cases_df['path'].map(read_pgm)

# remove the extracted files

!rm *.pgm

# show a sample

all_cases_df.sample(1)
import h5py

from tqdm import tqdm

from warnings import warn

def write_df_as_hdf(out_path, out_df):

    with h5py.File(out_path, 'w') as h:

        for k, arr_dict in tqdm(out_df.to_dict().items()): 

            try:

                s_data = np.stack(arr_dict.values(), 0)



                try:

                    h.create_dataset(k, data = s_data, compression = 'gzip')

                except TypeError as e: 

                    try:

                        h.create_dataset(k, data = s_data.astype(np.string_))

                    except TypeError as e2: 

                        print('%s could not be added to hdf5, %s' % (k, repr(e), repr(e2)))



            except ValueError as e:

                print('%s could not be created, %s' % (k, repr(e)))

                all_shape = [np.shape(x) for x in arr_dict.values()]

write_df_as_hdf('all_mias_scans.h5', all_cases_df)
# check the hdf5 file

print('Filesize: %2.2f mb' % (os.stat('all_mias_scans.h5').st_size/1e6))

with h5py.File('all_mias_scans.h5', 'r') as f:

    for k,v in f.items():

        print(k, v.shape)
f_stack = np.stack([x.astype(np.float32)[::2, ::2] for x in all_cases_df['scan'].values],0)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 8))

ax1.imshow(np.median(f_stack,0), cmap = 'magma')

ax1.axis('off')

ax2.imshow(np.std(f_stack,0), cmap = 'magma')

ax2.axis('off')

ax3.imshow(np.max(f_stack,0), cmap = 'magma')

ax3.axis('off')

fig.savefig('stack_images.pdf')
from skimage.util.montage import montage2d

fig, ax1 = plt.subplots(1,1, figsize = (12, 12))

ax1.imshow(montage2d(f_stack), cmap = 'magma')

fig.savefig('montage.png')