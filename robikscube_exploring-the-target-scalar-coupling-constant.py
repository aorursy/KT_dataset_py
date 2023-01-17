import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pylab as plt

plt.style.use('ggplot')

color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

structures = pd.read_csv('../input/structures.csv')

ss = pd.read_csv('../input/sample_submission.csv')
# Distribution of the target

train_df['scalar_coupling_constant'].plot(kind='hist', figsize=(20, 5), bins=1000, title='Distribution of the scalar coupling constant (target)')

plt.show()
color_index = 0

axes_index = 0

fig, axes = plt.subplots(8, 1, figsize=(20, 30), sharex=True)

for mtype, d in train_df.groupby('type'):

    d['scalar_coupling_constant'].plot(kind='hist',

                  bins=1000,

                  title='Distribution of Distance Feature for {}'.format(mtype),

                  color=color_pal[color_index],

                  ax=axes[axes_index])

    if color_index == 6:

        color_index = 0

    else:

        color_index += 1

    axes_index += 1

plt.show()
train_df.groupby('type')['scalar_coupling_constant'].plot(kind='hist',

                                                          bins=1000,

                                                          figsize=(20, 5),

                                                          alpha=0.8,

                                                         title='scalar_coupling_constant by coupling type')

plt.legend()

plt.show()