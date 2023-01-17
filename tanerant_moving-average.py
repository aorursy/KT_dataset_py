# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/tesla-stock-data-from-2010-to-2020/TSLA.csv')

df.tail()
import matplotlib.pyplot as plt

import seaborn as sns
close = df['Adj Close']
close.head()
close.plot()
close.index = pd.DatetimeIndex(close.index)
%config InlineBackend.figure_format = 'retina'

sns.set_context("poster")

sns.set(rc={'figure.figsize': (16, 9.)})

sns.set_style("whitegrid")
close.plot(style="-")
seven = close.rolling(window = 7).mean()

ten = close.rolling(window = 10).mean()

twelve = close.rolling(window = 20).mean()

close.plot(style = "-")

seven.plot(style = "--")

ten.plot(style = "-.")

twelve.plot(style = ":")

plt.legend(["input", "seven","ten","twelve"], loc = "upper left");

seven.tail()