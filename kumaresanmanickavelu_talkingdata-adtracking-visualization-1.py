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
train_sample = pd.read_csv('../input/train_sample.csv')
train_sample.head()
train_sample.describe()
train_sample.is_attributed.plot.hist()
train_good = train_sample[(train_sample.is_attributed == 1)]
train_bad = train_sample[(train_sample.is_attributed == 0)]
train_good.describe()
train_bad.describe()
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(2, 3,figsize=(20, 5))
train_sample.app.plot.hist(ax=axarr[0,0],title='train_sample')
train_good.app.plot.hist(ax=axarr[0,1],title='train_good')
train_bad.app.plot.hist(ax=axarr[0,2],title='train_bad')

train_sample[train_sample.app > 50].app.plot.hist(ax=axarr[1,0],title='train_sample > 50')
train_good[train_good.app > 50].app.plot.hist(ax=axarr[1,1],title='train_good > 50')
train_bad[train_bad.app > 50].app.plot.hist(ax=axarr[1,2],title='train_bad > 50')
fig, axarr = plt.subplots(2, 3,figsize=(20, 5))
train_sample.ip.plot.hist(ax=axarr[0,0],title='train_sample')
train_good.ip.plot.hist(ax=axarr[0,1],title='train_good')
train_bad.ip.plot.hist(ax=axarr[0,2],title='train_bad')

train_sample[train_sample.ip > 50].os.plot.hist(ax=axarr[1,0],title='train_sample > 50')
train_good[train_good.ip > 50].os.plot.hist(ax=axarr[1,1],title='train_good > 50')
train_bad[train_bad.ip > 50].os.plot.hist(ax=axarr[1,2],title='train_bad > 50')
fig, axarr = plt.subplots(1, 3,figsize=(20, 5))
train_sample.device.plot.hist(ax=axarr[0],title='train_sample')
train_good.device.plot.hist(ax=axarr[1],title='train_good')
train_bad.device.plot.hist(ax=axarr[2],title='train_bad')
fig, axarr = plt.subplots(2, 3,figsize=(20, 5))
train_sample.os.plot.hist(ax=axarr[0,0],title='train_sample')

train_good.os.plot.hist(ax=axarr[0,1],title='train_good')
train_bad.os.plot.hist(ax=axarr[0,2],title='train_bad')

train_sample[train_sample.os > 100].os.plot.hist(ax=axarr[1,0],title='train_sample > 100')
train_good[train_good.os > 100].os.plot.hist(ax=axarr[1,1],title='train_good > 100')
train_bad[train_bad.os > 100].os.plot.hist(ax=axarr[1,2],title='train_bad > 100')


fig, axarr = plt.subplots(1, 3,figsize=(20, 5))
train_sample.channel.plot.hist(ax=axarr[0],title='train_sample')
train_good.channel.plot.hist(ax=axarr[1],title='train_good')
train_bad.channel.plot.hist(ax=axarr[2],title='train_bad')
