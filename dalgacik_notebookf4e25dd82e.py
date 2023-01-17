# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.io import loadmat

import glob

from os.path import split, splitext
files_df = pd.DataFrame([splitext(split(x)[1])[0].split('_') for x in glob.glob('../input/train_1/*')]).astype('int')

files_df.columns = ['subject', 'segment', 'label']
file_list = glob.glob('../input/train_1/*')
mat0 = loadmat(file_list[0])
mat0