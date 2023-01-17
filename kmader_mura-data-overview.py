import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

# io related

from skimage.io import imread

import os

from glob import glob

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 
all_scans_df = pd.DataFrame(dict(path = 

                                 glob(os.path.join('..', 'input', '*', '*', '*', '*', '*', '*.png'))))

all_scans_df['TrainSplit'] = all_scans_df['path'].map(lambda x: x.split('/')[2])

all_scans_df['Region'] = all_scans_df['path'].map(lambda x: x.split('/')[-4])

all_scans_df['Patient'] = all_scans_df['path'].map(lambda x: x.split('/')[-3])

all_scans_df['FolderId'] = all_scans_df['path'].map(lambda x: x.split('/')[-2])

all_scans_df['Study'] = all_scans_df['FolderId'].map(lambda x: x.split('_')[0])

all_scans_df['Label'] = all_scans_df['FolderId'].map(lambda x: x.split('_')[-1] if '_' in x else np.NAN )

all_scans_df.sample(20)
all_scans_df['Region'].hist()

plt.xticks(rotation = 90)
sub_df = all_scans_df.groupby(['Region', 'Label']).apply(lambda x: x.sample(1)).reset_index(drop = True)

fig, (m_axs) = plt.subplots(4, sub_df.shape[0]//4, figsize = (12, 12))

for c_ax, (_, c_row) in zip(m_axs.flatten(), sub_df.iterrows()):

    c_ax.imshow(imread(c_row['path']), cmap = 'bone')

    c_ax.axis('off')

    c_ax.set_title('{Region}:{Label}'.format(**c_row))

fig.savefig('samples.png', dpi = 300)
all_scans_df.to_csv('all_cases.csv', index=False)