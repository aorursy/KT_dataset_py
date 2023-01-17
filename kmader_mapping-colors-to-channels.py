import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import matplotlib.pyplot as plt # making plots

from skimage.util.montage import montage2d # showing a montage

import os

import seaborn as sns

from warnings import warn

base_path = '../input/MultiSpectralImages'
all_files = glob(os.path.join(base_path, '*'))

label_df = pd.read_csv(os.path.join(base_path, 'Labels.csv'))

label_df = label_df[['Label', 'FileName']] # other columns are NaNs

label_df['Number'] = label_df['Label'].map(lambda x: int(x.split(' ')[-1]))

label_df['Color'] = label_df['Label'].map(lambda x: x.split(' ')[0])

print('Number of numbers',label_df.shape)

label_df.sample(3)
label_df['Number'].plot.hist()
%%time

out_results = []

for i, i_row in label_df.iterrows():

    cur_image_path = os.path.join(base_path, i_row['FileName'])

    if os.path.exists(cur_image_path):

        cur_df = pd.read_csv(cur_image_path)

        cur_df = cur_df[cur_df.columns[:-1]] # drop the last column

        out_results += [dict(list(i_row.items())+

             list(dict(cur_df.query('Channel0<255').apply(np.mean,axis = 0)).items()))]

    else:

        warn('File is missing {}'.format(cur_image_path), RuntimeWarning)



summary_df = pd.DataFrame(out_results)

summary_df.sample(3)
sns.pairplot(summary_df, hue = 'Color', vars = ['Channel{}'.format(i) for i in range(10)])
sns.pairplot(summary_df, hue = 'Number', vars = ['Channel{}'.format(i) for i in range(10)])