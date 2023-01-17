# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import gc #garbage collection

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

from datetime import datetime, timedelta, date # handling dates

from tqdm.notebook import tqdm # progress bars



# LightGBM

import lightgbm as lgb



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# Do not truncate view when max_cols is exceeded

# ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html

pd.set_option('display.max_columns', 50) 



# Path to Data Folder

KAGGLE_DATA_FOLDER = '/kaggle/input/m5-forecasting-accuracy'



# Paths to Models per Category

PATH_TO_CAT0="/kaggle/input/models-per-cat-with-sale/model_cat0_v10.lgb"

PATH_TO_CAT1="/kaggle/input/models-per-cat-with-sale/model_cat1_v10.lgb"

PATH_TO_CAT2="/kaggle/input/models-per-cat-with-sale/model_cat2_v10.lgb"



# Path to Model over All Categories

PATH_SINGLE_MODEL = "/kaggle/input/lgbmindividualbestsubmission/model_v13_param_tuning.lgb"

model_all = lgb.Booster(model_file = PATH_SINGLE_MODEL)

model_cat0 = lgb.Booster(model_file = PATH_TO_CAT0)  # Hobbies

model_cat1 = lgb.Booster(model_file = PATH_TO_CAT1)  # Household

model_cat2 = lgb.Booster(model_file = PATH_TO_CAT2)  # Foods

cat_models = [model_cat0, model_cat1, model_cat2]

model_names = ['hobby','household', 'food']
plt.rcParams['figure.figsize'] = (18.0, 4)

%matplotlib inline 



for category, cat_name in enumerate(model_names):

    fig, ax = plt.subplots(figsize=(12,8))

    lgb.plot_importance(cat_models[category], max_num_features=50, height=0.8, ax=ax)

    ax.grid(False)

    plt.title(f"Feature importance (#splits) for {cat_name} products", fontsize=15)

    plt.show()
plt.rcParams['figure.figsize'] = (18.0, 4)

%matplotlib inline 



for category, cat_name in enumerate(model_names):

    fig, ax = plt.subplots(figsize=(12,8))

    lgb.plot_importance(cat_models[category], max_num_features=50, height=0.8, ax=ax, importance_type='gain')

    ax.grid(False)

    plt.title(f"Feature importance (total gain) for {cat_name} products", fontsize=15)

    plt.show()
plt.rcParams['figure.figsize'] = (18.0, 4)

%matplotlib inline 



fig, ax = plt.subplots(figsize=(12,8))

lgb.plot_importance(model_all, max_num_features=50, height=0.8, ax=ax)

ax.grid(False)

plt.title(f"Feature importance (#splits)", fontsize=15)

plt.show()
fig, ax = plt.subplots(figsize=(12,8))

lgb.plot_importance(model_all, max_num_features=50, height=0.8, ax=ax, importance_type='gain')

ax.grid(False)

plt.title(f"Feature importance (total gain)", fontsize=15)

plt.show()