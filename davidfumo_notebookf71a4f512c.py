# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

import seaborn as sns



% matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

macro_df = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])



df_train = pd.merge(train_df, macro_df, how='left', on='timestamp')

df_test = pd.merge(test_df, macro_df, how='left', on='timestamp')
df_train.head()
df_train.shape
df_train.dtypes