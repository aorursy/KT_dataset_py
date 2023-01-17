# 基本ライブラリ

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

INPUT_PATH = Path("../input/liverpool-ion-switching/")

train = pd.read_csv(INPUT_PATH/"train.csv")

test = pd.read_csv(INPUT_PATH/"test.csv")

sub = pd.read_csv(INPUT_PATH/"sample_submission.csv")
train.shape, test.shape, sub.shape
plt.figure(figsize=(25, 6))

train.signal.plot()

plt.title("Train data")
plt.figure(figsize=(25, 6))

train.open_channels.plot(color="red")

plt.title("Target")
plt.figure(figsize=(25, 6))

test.signal.plot(color="green")

plt.title("Test data")
plt.figure(figsize=(25, 6))

sns.countplot(data=train, x="open_channels")

plt.title("Countplot of target data")
tab_df = pd.crosstab(train.open_channels, np.round(train.signal),)
tab_df
plt.figure(figsize=(23, 12))

sns.heatmap(tab_df, annot=True, fmt="d")