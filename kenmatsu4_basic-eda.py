# Basic library

import pandas as pd

import pandas.io.sql as psql

import numpy as np

import numpy.random as rd

import gc

import multiprocessing as mpa

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



# Visualization

import matplotlib

from matplotlib import font_manager

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from matplotlib import rc



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



pd.set_option("display.max_colwidth", 100)

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)

pd.options.display.float_format = '{:,.5f}'.format



%matplotlib inline

#%config InlineBackend.figure_format='retina'

print(os.listdir("../input"))
HOME_PATH = Path("../")

INPUT_PATH = Path(HOME_PATH/"input")

SAVE_PATH = Path(HOME_PATH/f"processed")

SAVE_PATH.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(INPUT_PATH/"train.csv", index_col=0)

idx_pos = train[train.target==1].index.values

idx_neg = train[train.target==0].index.values



test = pd.read_csv(INPUT_PATH/"test.csv", index_col=0)

sample_sub = pd.read_csv(INPUT_PATH/"sample_submission.csv", index_col=0)



target = train.target.values

del train["target"]



df_all = pd.concat([train, test],axis=0)
train.shape, test.shape, sample_sub.shape
train.head()
test.head()
sample_sub.head()
feat_name, nunique, std = [], [], []

for c in df_all.columns:

    #print(c, df_all[c].nunique())

    feat_name.append(c)

    nunique.append(df_all[c].nunique())

    std.append(df_all[c].std())

df_unique = pd.DataFrame({"feat": feat_name, "n_uniq": nunique, "std_": std})
df_unique.sort_values("n_uniq").head(30)
plt.figure(figsize=(10,4))

df_unique.plot.scatter(x="n_uniq", y="std_")
sns.countplot(x="target", data=pd.DataFrame(target, columns=["target"]))

pd.Series(target).value_counts()
train_corr = train.corr()
plt.figure(figsize=(12,10))

sns.heatmap(train_corr)
test_corr = test.corr()
plt.figure(figsize=(12,10))

sns.heatmap(test_corr)
corr_list = []

for i, c in enumerate(train.columns):

    corr = pd.DataFrame({f"{c}":train[c], "target":target}).corr().iloc[0,1]

    corr_list.append(corr)

    #print(f"{c},corr: {corr:.4f}")
idx_good_corr = np.argsort(np.abs(corr_list))[::-1]
df_good_corr_cols = pd.DataFrame({"ID":train.columns[idx_good_corr], "corr":np.array(corr_list)[idx_good_corr]}).head(50)

df_good_corr_cols
plt.figure(figsize=(24,5))

sns.barplot(x="col", y="corr", data=pd.DataFrame({"col":train.columns, "corr":corr_list}))

plt.xticks(rotation=90)

plt.show()
from itertools import combinations
comb_list = combinations(df_good_corr_cols.ID.tolist()[:15], 2)

#n_graph = len(comb_list)

plt.figure(figsize=(20,80))

for i, (c1, c2) in enumerate(comb_list):

    #print(c1, c2)

    df_tmp = pd.DataFrame({f"{c1}":train[c1], f"{c2}":train[c2], "target":target})

    

    plt.subplot(21,5,i+1)

    sns.scatterplot(x=f"{c1}", y=f"{c2}", hue="target", data=df_tmp.sample(frac=0.05), alpha=0.3)

    

    #break

plt.tight_layout()

plt.show()
plt.figure(figsize=(20,80))

for i, c in enumerate(train.columns):

    plt.subplot(40,5,i+1)

    

    max_, min_ = np.max(train[c]), np.min(train[c])

    bins = np.linspace(min_,max_,51)



    train[c].loc[idx_pos].hist(alpha=0.5,bins=bins, label="pos", density=True)

    train[c].loc[idx_neg].hist(alpha=0.5,bins=bins, label="neg", density=True)

    plt.legend(loc="best")

    plt.title(f"{c}")

plt.tight_layout()

plt.show()
plt.figure(figsize=(20,80))

for i, c in enumerate(train.columns):

    plt.subplot(40,5,i+1)

    max_, min_ = np.max([train[c].max(), test[c].max()]), np.min([train[c].min(), test[c].min()])

    bins = np.linspace(min_,max_,51)

    train[c].hist(alpha=0.5,bins=bins, label="train", density=True)

    test[c].hist(alpha=0.5,bins=bins, label="test", density=True)

    plt.legend(loc="best")

    plt.title(f"{c}")

plt.tight_layout()

plt.show()
from itertools import combinations
comb = combinations(train.columns, 2)

corr_list = []

feat_list = []

for c1, c2 in comb:

    div_feat1 = train[c1]/train[c2]

    corr = pd.DataFrame({f"{c1}_{c2}_div":div_feat1, "target":target}).corr().iloc[0,1]

    corr_list.append(corr)    

    feat_list.append(f"{c1}_{c2}_div")

    

    div_feat2 = train[c2]/train[c1]

    corr = pd.DataFrame({f"{c2}_{c1}_div":div_feat2, "target":target}).corr().iloc[0,1]

    corr_list.append(corr)    

    feat_list.append(f"{c2}_{c1}_div")

    

    mul_feat = train[c2]*train[c1]

    corr = pd.DataFrame({f"{c2}_{c1}_mul":mul_feat, "target":target}).corr().iloc[0,1]

    corr_list.append(corr)    

    feat_list.append(f"{c2}_{c1}_mul")
df_corr_feat = pd.DataFrame({"feat": feat_list,"corr": corr_list, "abs_corr": np.abs(corr_list)})

df_corr_feat.sort_values("abs_corr", ascending=False, inplace=True)
df_corr_feat.head(50)
plt.figure(figsize=(24,5))

sns.barplot(x="feat", y="corr", data=df_corr_feat.iloc[:100])

plt.xticks(rotation=90)

plt.show()
%%time

import hypertools as hyp

rd.seed(71)

idx = np.arange(train.shape[0])

rd.shuffle(idx)

hyp.plot(train.values[idx[:10000]], '.', reduce='TSNE', hue=target[idx[:10000]], ndims=2)