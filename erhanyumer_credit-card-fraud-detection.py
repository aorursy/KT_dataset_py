# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # ploting histograms, subplots etc...

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data  = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

data.head()
data.info()
data.corr()
f, ax = plt.subplots(figsize=(26, 26))

sns.heatmap(data. corr(), annot = True, linewidths = .5, fmt = '.1f', ax = ax)

plt.show()
data.head(10)
data.columns
data.Time.plot(kind = 'line', color = 'r', label = 'Time', linewidth = 1, alpha = 1, grid = True, linestyle = ':')

data.Amount.plot(kind = 'line', color = 'g', label = 'Amount', linewidth = 1, alpha = 1, grid = True, linestyle = '-')
data.Time.plot(kind = 'line', color = 'r', label = 'Time', linewidth = 1, alpha = 1, grid = True, linestyle = ':')

data.Amount.plot(kind = 'line', color = 'g', label = 'Amount', linewidth = 1, alpha = 1, grid = True, linestyle = '-')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data.plot(kind = 'scatter', x = 'Time', y = 'Amount', alpha = .5, color = 'r')

plt.xlabel('Time')

plt.ylabel('Amount')

plt.title('Time - Amount Scatter Plot')
data.Amount.plot(kind = 'hist', bins = 50, figsize = (12, 12))

plt.show()
data.Amount.plot(kind = 'hist', bins = 50)

plt.clf()