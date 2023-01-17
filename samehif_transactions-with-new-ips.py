# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train_day_7 = pd.read_csv('../input/train.csv', skiprows=range(1,9308568), nrows=59633310, usecols=['click_time', 'ip', 'is_attributed'])
train_day_8 = pd.read_csv('../input/train.csv', skiprows=range(1,68941878), nrows=62945075, usecols=['click_time', 'ip', 'is_attributed'])
print (train_day_7.shape, train_day_7.is_attributed.mean())
print (train_day_8.shape, train_day_8.is_attributed.mean())
train_day_7.ip.nunique()
train_day_8.ip.nunique()
train_day_8[~train_day_8.ip.isin(train_day_7.ip.unique())].ip.nunique()
train_day_8[~train_day_8.ip.isin(train_day_7.ip.unique())].shape[0] / float(train_day_8.shape[0])
train_day_8[~train_day_8.ip.isin(train_day_7.ip.unique())].is_attributed.mean()