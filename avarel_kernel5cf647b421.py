# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, Ridge, ElasticNet, ElasticNetCV # Lasso is L1, Ridge is L2, ElasticNet is both

from sklearn.model_selection import ShuffleSplit # For cross validation

from sklearn.cluster import KMeans

import lightgbm as lgb # LightGBM is an alternative to XGBoost. I find it to be faster, more accurate and easier to install.



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



train_df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')



# Any results you write to the current directory are saved as output.
train_df.head()
train_df["Original Release Date"] = pd.to_datetime(train_df["Original Release Date"])
train_df["Current Version Release Date"] = pd.to_datetime(train_df["Current Version Release Date"])
train_df.head()
import math

train_df[np.isnan(train_df["Average User Rating"])] = 0

train_df["Average User Rating"]
train_df["Size"]
plt.plot(train_df["Size"], train_df["Average User Rating"], alpha = 0.5)