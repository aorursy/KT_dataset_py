import pandas as pd

import pandas.io.sql as psql

import numpy as np

import numpy.random as rd

import gc

import multiprocessing as mp

import os

import sys

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



import pickle

def unpickle(filename):

    with open(filename, 'rb') as fo:

        p = pickle.load(fo)

    return p



def to_pickle(filename, obj):

    with open(filename, 'wb') as f:

        pickle.dump(obj, f, -1)



# pandas settings

pd.set_option("display.max_colwidth", 100)

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)

pd.options.display.float_format = '{:,.5f}'.format



# Graph drawing

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score

import seaborn as sns

import math
df = pd.read_csv("../input/liverpool-ion-switching/train.csv")

df_test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
df.shape, df_test.shape
n_unit = 50000

n_groups = int(len(df)/n_unit) #40

df["group"] = 0

for i in range(n_groups):

    ids = np.arange(i*n_unit, (i+1)*n_unit)

    df.loc[ids,"group"] = i
for i in range(n_groups):

    sub = df[df.group == i]

    signals = sub.signal.values

    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))

    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals) + 2)

#     signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals))

    signals = signals*(imax-imin)

    df.loc[sub.index,"open_channels_pred"] = np.array(signals,np.int)
df.head()
cross_df = pd.crosstab(df.open_channels, df.open_channels_pred)

cross_df
score = cohen_kappa_score(df.open_channels, df.open_channels_pred, weights = 'quadratic')

print(f"QWK: {score}")
plt.figure(figsize=(15,10))

sns.heatmap(cross_df, annot=True, fmt="d")