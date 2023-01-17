import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

# io related

from skimage.io import imread

import os

from glob import glob

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 
glob(os.path.join(data_dir, '*', '*'))
data_dir = os.path.join('..', 'input')

all_files = glob(os.path.join(data_dir, '*', '*'))

all_df = pd.DataFrame(dict(path = [x for x in all_files if x.endswith('png') or x.endswith('gui')]))

all_df['source'] = all_df['path'].map(lambda x: x.split('/')[-2])

all_df['filetype'] = all_df['path'].map(lambda x: os.path.splitext(x)[1][1:])

all_df['fileid'] = all_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])

all_df.sample(3)
all_data_df = all_df.pivot_table(index=['source', 'fileid'], 

                                 columns = ['filetype'], 

                                 values = 'path',

                                 aggfunc='first').reset_index()

print(all_data_df.shape[0], 'samples for training and validation')

all_data_df.sample(3)
def read_text_file(in_path):

    with open(in_path, 'r') as f:

        return f.read()

clear_sample_df = all_data_df.groupby(['source']).apply(lambda x: x.sample(1)).reset_index(drop = True)

fig, m_axs = plt.subplots(2, clear_sample_df.shape[0], figsize = (12, 8))

for (_, c_row), (im_ax, gui_ax) in zip(clear_sample_df.iterrows(), m_axs.T):

    im_ax.imshow(imread(c_row['png']))

    im_ax.axis('off')

    im_ax.set_title(c_row['source'])

    gui_ax.text(0, 0, read_text_file(c_row['gui']), 

            style='italic',

            bbox={'facecolor':'yellow', 'alpha':0.1, 'pad':10},

               fontsize = 7)

    gui_ax.axis('off')

fig.savefig('mapping.png', dpi = 300)