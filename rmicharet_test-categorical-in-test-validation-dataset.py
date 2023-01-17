# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%load_ext autoreload
%autoreload 2

%matplotlib inline
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
import datetime
from sklearn import metrics
PATH = "../input"
df_raw = pd.read_csv(f'{PATH}/train.csv', low_memory=False, 
                     parse_dates=["Open Date"])
df_test = pd.read_csv(f'{PATH}/test.csv', low_memory=False, 
                     parse_dates=["Open Date"])
train_cats(df_raw)
df_raw.Type.cat.categories
df_test.groupby(['Type'])['Type'].count()
arrId_MB = df_test[df_test['Type'].str.contains('MB')].Id
df_test_copia = df_test.copy()
apply_cats(df_test,df_raw)
df_test.Type.cat.categories
df_test.groupby(['Type'])['Type'].count()
df_test[df_test.Id.isin(arrId_MB)].head()
