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
df = pd.read_csv('../input/aapl123/aapl.csv',index_col='Date',parse_dates=True)
df.info()
#creating New Cloumn
df['Check'] = -9999
df.head()
df.tail()
df.drop(columns='Check')
list_keys = ['Country', 'Total']
list_values = [['United States', 'Soviet Union', 'United Kingdom'], [1118, 473, 273]]
# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys,list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
da = pd.DataFrame(data)
print(da)

sunspot = pd.read_csv('../input/sunspot/ISSN_D_tot.csv',header=None)
sunspot.head()
import matplotlib.pyplot as plt
close_series = df['Close']
type(close_series)
close_series.plot()
plt.show()
plt.plot(close_series)
plt.show()
df.plot()
plt.yscale('log')
plt.show()
plt.plot(df)
plt.yscale('log')
plt.show()
