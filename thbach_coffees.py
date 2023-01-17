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
data = pd.read_csv("../input/coffees.csv")
data
data.head()
# .loc or .iloc
data.iloc[2]
data.coffees[:5]
data.describe()
# .isnull() and boolean indexing with []
data[data.coffees.isnull()]
# .dtypes
data.dtypes
# print the first element of the series with [] indexsing
print(data.timestamp[0])

#print its type()
print(type(data.timestamp[0]))
# cast hte coffees column using pd.to_numeric and coerce erros
data.coffees = pd.to_numeric(data.coffees, errors="coerce")
data.head()
# data.dtypes
data.dropna(inplace=True)
# or data = data.dropna()
data.head()
data.coffees = data.coffees.astype(int)
data.head()
# data.dtypes
# pad.to_datetime()
data.timestamp = pd.to_datetime(data.timestamp)
data.head()
# data.dtypes
data.describe(include="all")
# whgat do the first few rows look like
data.iloc[:5]
# .plot() on the coffees series
data.coffees.plot()
# .plot() on the dataframe, setting x to the timestamp, with dot-dash style

data.timestamp = data.timestamp.astype(np.int64)
data.dtypes
# DataFrame.plot(x=None, y=None, kind='line', ax=None, subplots=False, sharex=None, sharey=False, layout=None, figsize=None, use_index=True, title=None, grid=None, legend=True, style=None, logx=False, logy=False, loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None, table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False, **kwds)
data.plot(x=data.timestamp, y=data.coffees)



