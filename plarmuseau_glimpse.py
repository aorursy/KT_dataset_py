# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt



tick = pd.read_csv('../input/supercolumns-elements-nasdaq-nyse-otcbb-general-UPDATE-2017-03-01.csv')

print('Size of tickers set: {} rows and {} columns'.format(*tick.shape))

print(tick.head(7))
stats=tick.describe().T

stats
tick[tick['concept:Lithium']>0].sort_values(by='concept:Lithium')
lico=tick.corr().sort_values(by='concept:Lithium')

print(lico.head())

import seaborn as sns

%matplotlib inline



# plot the heatmap

sns.heatmap(lico[lico['concept:Lithium']>0.25])
print(tick[tick['Symbol:update-2017-04-01:']=='TSLA'])

print(tick[tick['Symbol:update-2017-04-01:']=='NVDA'])


