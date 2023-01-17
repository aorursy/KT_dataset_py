# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# Any results you write to the current directory are saved as output.
ROOT = "/kaggle/input/trends-assessment-prediction/"
!ls {ROOT}
# image and mask directories
data_dir = f'{ROOT}/fMRI_train'
#!python
#!/usr/bin/env python
from scipy.io import loadmat
path = data_dir + '/10025.mat'
import h5py
with h5py.File(path, 'r') as file:
    print(list(file.keys()))
    print(file)
f = h5py.File(path, 'r')
f['SM_feature']
f['SM_feature'][0][0]