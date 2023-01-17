# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas_profiling

import pandas_summary as ps



# Data processing, metrics and modeling

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import TruncatedSVD, PCA

from sklearn.impute import SimpleImputer



# Lgbm

import lightgbm as lgb



# Suppr warning

import warnings

warnings.filterwarnings("ignore")



# Plots

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams



# Others

import shap

import datetime

from tqdm import tqdm_notebook

import sys

import pickle

import re

import json

import gc



pd.set_option('display.max_columns', 5000)

pd.set_option('display.max_rows', 500)

pd.set_option('display.width', 1000)

pd.set_option('use_inf_as_na', True)



warnings.simplefilter('ignore')

matplotlib.rcParams['figure.dpi'] = 100

sns.set()

%matplotlib inline
folder = '/kaggle/input/find-a-defect-in-the-production-extrusion-line/'

stats_df = pd.read_csv(folder + 'stat.csv', sep=',')

full_df = pd.read_csv(folder + 'extrusion.csv', sep=',')
stats_df.shape, full_df.shape
stats_df.head()
full_df.head()
full_df.tail()
full_df['Datum'] = pd.to_datetime(full_df['Datum'])

full_df = full_df.reset_index()

full_df.index = full_df['Datum'] 
full_df.head()
dfs = ps.DataFrameSummary(full_df)

dfs.summary()
stats_df[(stats_df['Tags'] == 'ST110_VAREx_0_SDickeIst')]
full_df['ST110_VAREx_0_SDickeIst'].hist(bins=40);
full_df[full_df['ST110_VAREx_0_SDickeIst'] < 50]['ST110_VAREx_0_SDickeIst'].hist(bins=40);
full_df[(full_df['ST110_VAREx_0_SDickeIst'] < 28)]['ST110_VAREx_0_SDickeIst'].shape[0]
full_df[(full_df['ST110_VAREx_0_SDickeIst'] > 0)].shape[0]
full_df['ST110_VAREx_0_SDickeIst'].apply(lambda x: 1 if x == 0 else 0).value_counts(normalize=True)
full_df['ST110_VAREx_0_SDickeIst'].apply(lambda x: 1 if x == 0 else 0).hist();
full_df[['ST110_VAREx_0_SDickeIst']].describe()