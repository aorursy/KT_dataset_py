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
# seaborn plotting library

import seaborn as sns

# load basic style defaults

sns.set()
# set a random seed for reproducibility

np.random.seed(1)

# generate an array of 10,000 normally distributed random numbers

x = np.random.randn(10000)

# convert the array to a pandas series object

x = pd.Series(x, name="x Variable")

# view summary statistics for the series

x.describe()
ax = sns.distplot(x)
ax = sns.distplot(x, kde=False)
from scipy.stats import norm, expon, cauchy

# plot with a normal distribution fit

ax = sns.distplot(x, fit=norm, kde=False)
# plot with an exponential distribution fit (not very sensible here)

ax = sns.distplot(x, fit=expon, kde=False)
# plot with a normal distribution fit (not completely unreasonable, but clearly inferior to normal fit)

ax = sns.distplot(x, fit=cauchy, kde=False)
# using only the first 100 points for clarity 

ax = sns.distplot(x[0:100], rug=True)
ax = sns.distplot(x, vertical=True)
# adjust how color codes are interpreted

sns.set_color_codes()

# produce a plot with a green color palette

ax = sns.distplot(x, color="g")
# define formatting dictionaries

hist_kws={"histtype": "step",

          "linewidth": 3, 

          "alpha": 0.8, 

          "color": "c"}



kde_kws={"color": "y", 

         "lw": 4, 

         "label": "KDE"}



rug_kws={"color": "m"}



# again using only the first 100 points for clarity on rug plot

ax = sns.distplot(x[0:100], rug=True, rug_kws=rug_kws, kde_kws=kde_kws, hist_kws=hist_kws)