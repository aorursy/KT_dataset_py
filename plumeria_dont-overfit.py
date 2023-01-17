

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import gc,os,sys

sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format

print(os.listdir("../input"))
test=pdtest = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

train.head(10)
test.head(10)
for c in train.columns:

    if c not in test.columns: print(c)
null_cnt = train.isnull().sum().sort_values()

print('null count:', null_cnt[null_cnt > 0])
train['target'].value_counts().to_frame().plot.bar()