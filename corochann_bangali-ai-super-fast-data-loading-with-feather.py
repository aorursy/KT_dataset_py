import gc

import os

from pathlib import Path

import random

import sys



from tqdm import tqdm_notebook as tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)
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
%%time

datadir = Path('/kaggle/input/bengaliai-cv19')



# Read in the data CSV files

train = pd.read_csv(datadir/'train.csv')

test = pd.read_csv(datadir/'test.csv')

sample_submission = pd.read_csv(datadir/'sample_submission.csv')

class_map = pd.read_csv(datadir/'class_map.csv')
%%time

train_image_df0 = pd.read_parquet(datadir/'train_image_data_0.parquet')
%%time

featherdir = Path('/kaggle/input/bengaliaicv19feather')



train_image_df0 = pd.read_feather(featherdir/'train_image_data_0.feather')

train_image_df1 = pd.read_feather(featherdir/'train_image_data_1.feather')

train_image_df2 = pd.read_feather(featherdir/'train_image_data_2.feather')

train_image_df3 = pd.read_feather(featherdir/'train_image_data_3.feather')
%%time

# Please change this to `True` when actual submission

submission = False



if submission:

    test_image_df0 = pd.read_parquet(datadir/'test_image_data_0.parquet')

    test_image_df1 = pd.read_parquet(datadir/'test_image_data_1.parquet')

    test_image_df2 = pd.read_parquet(datadir/'test_image_data_2.parquet')

    test_image_df3 = pd.read_parquet(datadir/'test_image_data_3.parquet')

else:

    test_image_df0 = pd.read_feather(featherdir/'test_image_data_0.feather')

    test_image_df1 = pd.read_feather(featherdir/'test_image_data_1.feather')

    test_image_df2 = pd.read_feather(featherdir/'test_image_data_2.feather')

    test_image_df3 = pd.read_feather(featherdir/'test_image_data_3.feather')
train_image_df0.head()