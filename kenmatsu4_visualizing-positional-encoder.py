# 基本ライブラリ

import pandas as pd

import pandas.io.sql as psql

import numpy as np

import numpy.random as rd

import gc

import multiprocessing as mp

import os

import sys

import pickle

from collections import defaultdict

from glob import glob

import math

from datetime import datetime as dt

from pathlib import Path

import scipy.stats as st

import re

import shutil

from tqdm import tqdm_notebook as tqdm

import datetime

ts_conv = np.vectorize(datetime.datetime.fromtimestamp) # 秒ut(10桁) ⇒ 日付



# pandas settings

pd.set_option("display.max_colwidth", 100)

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)

pd.options.display.float_format = '{:,.5f}'.format



# グラフ描画系

import matplotlib

from matplotlib import font_manager

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from matplotlib import rc

from matplotlib_venn import venn2, venn2_circles

from matplotlib import animation as ani

from IPython.display import Image



plt.rcParams["patch.force_edgecolor"] = True

#rc('text', usetex=True)

from IPython.display import display # Allows the use of display() for DataFrames

import seaborn as sns

sns.set(style="whitegrid", palette="muted", color_codes=True)

sns.set_style("whitegrid", {'grid.linestyle': '--'})

red = sns.xkcd_rgb["light red"]

green = sns.xkcd_rgb["medium green"]

blue = sns.xkcd_rgb["denim blue"]



%matplotlib inline

%config InlineBackend.figure_format='retina'
def positional_encoder(pos, dim, dim_model):

    if dim%2==0:

        return np.sin(pos/10000**(dim/dim_model))

    else:

        return np.cos(pos/10000**((dim-1)/dim_model))
n_pos = 100

n_dim = 300

pe_val = np.zeros((n_dim, n_pos)).astype(np.float128)

for pos in range(n_pos):

    for dim in range(n_dim):

        pe_val[dim, pos] = positional_encoder(pos, dim, n_dim)
plt.figure(figsize=(25,10))

sns.heatmap(pe_val)

plt.xlabel("position")

plt.ylabel("dimension")
plt.figure(figsize=(25,10))

sns.heatmap(pe_val[::2,:])
plt.figure(figsize=(25,10))

sns.heatmap(pe_val[1::2,:])
for i in range(30):

    plt.figure(figsize=(15,3))

    plt.plot(pe_val[:,i])

    plt.title(i)

    plt.show()
np.corrcoef(pe_val[i,:], pe_val[j,:])
corr = np.zeros((n_pos, n_pos))

for i in range(n_pos):

    for j in range(n_pos):

        corr[i, j] = np.corrcoef(pe_val[i,:], pe_val[j,:])[0,1]

        

plt.figure(figsize=(15,13))

sns.heatmap(corr)
corr_triu = np.triu(corr)

for i in range(n_pos):

    corr_triu[i,i] = -1
plt.figure(figsize=(15,13))

sns.heatmap(corr_triu)
# つまり、自分以外は線形独立だということでいいのかな。

np.max(corr_triu)