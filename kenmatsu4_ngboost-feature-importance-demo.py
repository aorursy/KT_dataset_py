!pip install git+https://github.com/matsuken92/ngboost.git@feature/add_best_iteration
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

from ngboost.distns import Normal, LogNormal, Exponential



# skleran

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
data = load_boston()

X, y = data.data, data.target

rd.seed(71)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
%%time

rd.seed(71)

ngb = NGBoost(Base=default_tree_learner, Dist=Normal, #Normal, LogNormal

              Score=MLE(), natural_gradient=True, verbose=True,)

ngb.fit(X_train, y_train, X_val=X_valid, Y_val=y_valid)



y_preds = ngb.predict(X_valid)

dist_obj = ngb.pred_dist(X_valid)



# test Mean Squared Error

test_MSE = mean_squared_error(y_preds, y_valid)

print('ngb Test MSE', test_MSE)



#test Negative Log Likelihood

test_NLL = -dist_obj.logpdf(y_valid.flatten()).mean()

print('ngb Test NLL', test_NLL)
feature_importance = ngb.feature_importance()

feature_importance_df = pd.DataFrame(feature_importance, index=data.feature_names)
feature_importance_df