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
# Carregar os arquivos

train = pd.read_csv('../input/train.csv', parse_dates=[0])

test = pd.read_csv('../input/test.csv', parse_dates=[0])



train.shape, test.shape
train['count'] = np.log(train['count'])
df = pd.concat([train, test], sort=False)
df.info()
df.loc[2]
# Resetando o indice

df = df.reset_index(drop=True)
df.loc[2]
df['year'] = df.datetime.dt.year

df['month'] = df.datetime.dt.month

df['weekday'] = df.datetime.dt.weekday

df['day'] = df.datetime.dt.day

df['hour'] = df.datetime.dt.hour
train = df[~df['count'].isnull()]

test = df[df['count'].isnull()]
train.shape, test.shape
from sklearn.model_selection import train_test_split



train, valid = train_test_split(train, random_state=42)



train.shape, valid.shape
removed_col = ['count', 'casual', 'registered', 'datetime']

feats = [c for c in train.columns if c not in removed_col]