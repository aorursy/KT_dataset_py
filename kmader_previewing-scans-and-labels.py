# extract the files

!tar -xf ../input/all-mias.tar.gz
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
# load a single image

plt.imshow(read_pgm(glob('*.pgm')[0]))
all_cases_df = pd.read_table('../input/Info.txt', delimiter=' ')

all_cases_df = all_cases_df[all_cases_df.columns[:-1]] # drop last column

all_cases_df['path'] = all_cases_df['REFNUM'].map(lambda x: '%s.pgm' % x)

all_cases_df.sample(3)
all_cases_df['CLASS'].value_counts()
sample_count = 3

fig, m_axs = plt.subplots(len(all_cases_df['CLASS'].value_counts()), 3, figsize = (12, 20))

for c_axs, (c_cat, c_df) in zip(m_axs, all_cases_df.groupby('CLASS')):

    for c_ax, (_, c_row) in zip(c_axs, c_df.sample(sample_count).iterrows()):

        c_ax.imshow(read_pgm(c_row['path']), cmap = 'bone')

        c_ax.axis('off')

        c_ax.set_title('{CLASS}-{SEVERITY}'.format(**c_row))

fig.savefig('overview.pdf')