!pip install ngboost
# basic libraries

import pandas as pd

import numpy as np

import numpy.random as rd

import gc

import multiprocessing as mp

import os

import sys

import pickle

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



# グラフ描画系

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



# pandas formatting

pd.set_option("display.max_colwidth", 100)

pd.set_option("display.max_rows", None)

pd.set_option("display.max_columns", None)

pd.options.display.float_format = '{:,.5f}'.format



%matplotlib inline

%config InlineBackend.figure_format='retina'
# ngboost

from ngboost.ngboost import NGBoost

from ngboost.learners import default_tree_learner

from ngboost.scores import MLE

from ngboost.distns import Normal, LogNormal



# skleran

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



# lightgbm

import lightgbm as lgb
X, y = load_boston(True)

rd.seed(71)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
%%time

lgb_train = lgb.Dataset(X_train, y_train)

lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

model = lgb.train({'objective': 'regression', 

           'metric': "mse",

           'learning_rate': 0.01,

           'seed': 71},

            lgb_train, 

            num_boost_round=99999,

            valid_sets=[lgb_valid], 

            early_stopping_rounds=100, 

            verbose_eval=500)



y_pred_lgb = model.predict(data=X_valid)
%%time

rd.seed(71)

ngb = NGBoost(Base=default_tree_learner, Dist=Normal, #Normal, LogNormal

              Score=MLE(), natural_gradient=True, verbose=False, )

ngb.fit(X_train, y_train, X_val=X_valid, Y_val=y_valid)



y_preds = ngb.predict(X_valid)

y_dists = ngb.pred_dist(X_valid)





# test Mean Squared Error

test_MSE = mean_squared_error(y_preds, y_valid)

print('ngb Test MSE', test_MSE)



#test Negative Log Likelihood

test_NLL = -y_dists.logpdf(y_valid.flatten()).mean()

print('ngb Test NLL', test_NLL)
offset = np.ptp(y_preds)*0.1

y_range = np.linspace(min(y_valid)-offset, max(y_valid)+offset, 200).reshape((-1, 1))

dist_values = y_dists.pdf(y_range).transpose()



plt.figure(figsize=(25, 120))

for idx in tqdm(np.arange(X_valid.shape[0])):

    

    plt.subplot(35, 3, idx+1)

    plt.plot(y_range, dist_values[idx])

    

    plt.vlines(y_preds[idx], 0, max(dist_values[idx]), "r", label="ngb pred")

    plt.vlines(y_pred_lgb[idx], 0, max(dist_values[idx]), "purple", label="lgb pred")

    plt.vlines(y_valid[idx], 0, max(dist_values[idx]), "pink", label="ground truth")

    plt.legend(loc="best")

    plt.title(f"idx: {idx}")

    plt.xlim(y_range[0], y_range[-1])

plt.tight_layout()

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(x=y_preds, y=y_pred_lgb, s=20)

plt.plot([8,50], [8,50], color="gray", ls="--")

plt.xlabel("NGBoost")

plt.ylabel("LightGBM")

plt.title("NGBoost vs LightGBM")

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(y=y_preds, x=y_valid, s=20)

plt.plot([8,50], [8,50], color="gray", ls="--")

plt.ylabel("NGBoost")

plt.xlabel("Ground truth")

plt.title("NGBoost vs Ground truth")

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(y=y_pred_lgb, x=y_valid, s=20)

plt.plot([8,50], [8,50], color="gray", ls="--")

plt.ylabel("LightGBM")

plt.xlabel("Ground truth")

plt.title("LightGBM vs Ground truth")

plt.show()