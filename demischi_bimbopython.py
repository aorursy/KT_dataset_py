# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import display_markdown as mkdown # as print

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv('../input/train.csv', nrows=500000)
df_test = pd.read_csv('../input/test.csv', nrows=500000)

print('Size of training set: ' + str(df_train.shape))
print(' Size of testing set: ' + str(df_test.shape))

print('Columns in train: ' + str(df_train.columns.tolist()))
print(' Columns in test: ' + str(df_test.columns.tolist()))

print(df_train.describe())