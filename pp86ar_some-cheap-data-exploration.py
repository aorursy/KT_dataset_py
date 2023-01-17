

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')



n_bins = 20

for i in df_train.columns[0:]:

  print("Train Dataset")

  fig, axs = plt.subplots(1, 2, figsize = (15,5), sharey=True,)

  fig.suptitle('{} by Sex'.format(i))

  axs[0].hist(df_train[i][df_train.Sex == 'Male'], bins=n_bins)

  axs[0].set_title('Male')

  axs[1].hist(df_train[i][df_train.Sex == 'Female'], bins=n_bins)

  axs[1].set_title('Female')

  plt.show()