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
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
loan_df = pd.read_csv('../input/lending-club-loan-data/loan.csv', low_memory=False)

desc_df = pd.read_excel("../input/lending-club-loan-data/LCDataDictionary.xlsx")
loan_df.shape
loan_df.head()
loan_df.loan_status.value_counts()
loan_df_ = pd.concat([loan_df[loan_df.loan_status=="Fully Paid"].head(2), 

                      loan_df[loan_df.loan_status=="Charged Off"].head(2), 

                      loan_df[loan_df.loan_status=="Default"].head(2), ], axis=0)
loan_df_[["loan_amnt", "annual_inc", "int_rate", "term", "grade", "emp_title", "emp_length", "home_ownership", "addr_state","last_pymnt_d", "loan_status"]]
desc_df.set_index("LoanStatNew").loc[["loan_amnt", "annual_inc", "int_rate", "term", "grade", "emp_title", "emp_length", "home_ownership", "addr_state","last_pymnt_d", "loan_status"]]