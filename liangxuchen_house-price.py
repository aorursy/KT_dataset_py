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
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.columns
miss_train = list()
cols = df_train.columns
total = len(df_train['Id'])
for col in cols:
    miss = df_train[col].isnull().sum()
    if miss > 0:
        print("{}: {} missing, {}%".format(col, miss, round(miss/total * 100, 2)))
        miss_train.append(col)
miss_test = list()
cols = df_test.columns
total = len(df_test['Id'])
for col in cols:
    miss = df_test[col].isnull().sum()
    if miss > 0:
        print("{}: {} missing, {}%".format(col, miss, round(miss/total * 100, 2)))
        miss_test.append(col)
df_train_desc = df_train.describe()
df_train_desc.columns
df_train_desc.head()
#### cols = df_train.columns
cate_col = list()
for col in cols:
    if col not in df_train_desc.columns:
        cate_col.append(col)
print(','.join(cate_col))
    