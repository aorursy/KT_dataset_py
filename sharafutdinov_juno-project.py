import os
print(os.listdir("../input"))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tables
from random import shuffle
from IPython.display import clear_output
from sklearn import metrics
from tqdm import tqdm
import time
import seaborn as sns

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%matplotlib inline

# Any results you write to the current directory are saved as output.
lpmt_hits = pd.read_hdf('../input/train_lpmt_hits.h5', mode='r')
print('Number of Features: ', lpmt_hits.shape[1])
print('Number of Objects: ', lpmt_hits.shape[0])
print(lpmt_hits.head())
lpmt_pos = pd.read_csv('../input/lpmt_pos.csv')
print('Number of Features: ', lpmt_pos.shape[1])
print('Number of Objects: ', lpmt_pos.shape[0])
print(lpmt_pos.head())
