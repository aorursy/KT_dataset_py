import numpy as np

import pandas as pd

import pylab

import calendar

from scipy import stats

import missingno as msno

from datetime import datetime

import matplotlib.pyplot as plt

import warnings

import seaborn as sn
macro = pd.read_csv("../input/macro.csv")

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head(4).transpose()
fig,axes = plt.subplots(ncols=2)

fig.set_size_inches(20, 10)

stats.probplot(train["price_doc"], dist='norm', fit=True, plot=axes[0])

stats.probplot(np.log1p(train["price_doc"]), dist='norm', fit=True, plot=axes[1])