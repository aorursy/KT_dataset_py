# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from cv2 import imread, createCLAHE # read and equalize images

from glob import glob

%matplotlib inline

import matplotlib.pyplot as plt
all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')

all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input', 'data','images*', '*', '*.png'))}

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

all_xray_df['Patient Age'] = all_xray_df['Patient Age'].astype(int)

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
sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 1e-1

sample_weights /= sample_weights.sum()

all_xray_df = all_xray_df.sample(18000, weights=sample_weights)



label_counts = all_xray_df['Finding Labels'].value_counts()[:15]

fig, ax1 = plt.subplots(1,1,figsize = (12, 8))

ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

ax1.set_xticks(np.arange(len(label_counts))+0.5)

_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
all_xray_df.shape
all_xray_df.to_csv("sample_18000.csv")