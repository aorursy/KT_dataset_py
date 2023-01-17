# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from pathlib import Path

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_dir = '../input/jpeg-isic2019-512x512'

meta = pd.read_csv(os.path.join(data_dir, 'train.csv'))

display(meta.head(3))

print('Diagnosis: ' , meta['diagnosis'].unique())

print('anatom site ', meta['anatom_site_general_challenge'].unique())

# meta_p = meta[meta['diagnosis'] == 'MEL']

meta_p = meta

def rename_df(x):

    if 'torso' in x:

        return 'torso'

    else:

        return x 

meta_p.loc[:, 'anatom_site_general_challenge'] = meta_p['anatom_site_general_challenge'].fillna('NaN')

meta_p.loc[:, 'anatom_site_general_challenge'] = meta_p['anatom_site_general_challenge'].apply(rename_df)

print('anatom site ', meta_p['anatom_site_general_challenge'].unique())

display(meta_p)
meta_p.to_csv('train.csv', index=False)