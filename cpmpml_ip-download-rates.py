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

import matplotlib.pyplot as plt
%matplotlib inline
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

train = pd.read_csv('../input/train.csv', dtype=dtypes, usecols=['ip', 'is_attributed'])
test = pd.read_csv('../input/test.csv', dtype=dtypes, usecols=['ip'])

df = train.groupby('ip').is_attributed.mean().to_frame().reset_index()

df.head()
df['roll'] = df.is_attributed.rolling(window=1000).mean()
plt.plot(df.ip, df.roll)
df1 = df[(df.ip >= 120000) & (df.ip <= 130000)]
plt.plot(df1.ip, df1.roll)
df1 = df[(df.ip >= 126000) & (df.ip <= 126700)]
plt.plot(df1.ip, df1.roll)
df1 = df[(df.ip >= 126400) & (df.ip <= 126500)]
plt.plot(df1.ip, df1.roll)
df1 = df[(df.ip >= 126415) & (df.ip <= 126425)]
plt.plot(df1.ip, df1.roll)
test.ip.max()

