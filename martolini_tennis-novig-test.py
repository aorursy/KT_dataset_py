# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import strptime

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df = pd.read_csv('../input/Data.csv', encoding='latin-1')

df = df[['ATP', 'Location', 'Tournament', 'Date', 'PSW', 'PSL']]

df.Date = df.Date.map(lambda x: strptime(x, '%d/%m/%Y'))

df.dropna(inplace=True)

df.count()
df['margin'] = 1 / df.PSW + 1 / df.PSL - 1

df.head()
df['novig_w'] = 2 * df.PSW / (2 - df.margin * df.PSW)

df['novig_l'] = 2 * df.PSL / (2 - df.margin * df.PSL)

df.head()
df['net'] = (df.novig_w - 2) * 100

df.net.mean()
grouped = df.groupby(df.Date.map(lambda x: x[0]))[['net', 'margin']].mean()

plt.plot(grouped.net)