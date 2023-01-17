# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import scipy.stats as stats

rvs = stats.norm(scale=1, loc=0).rvs(1000000)

sns.distplot(rvs)
def plot_normal_distribution(loc, scale):

    rvs = stats.norm(loc=loc, scale=scale).rvs(1000000)

    sns.distplot(rvs, hist=False, label="stddev=" + str(scale) + ", mean=" + str(loc))

    

plt.figure(figsize=(16, 6))    

plt.subplots

plot_normal_distribution(loc=0, scale=1)

plot_normal_distribution(loc=0, scale=3)

plot_normal_distribution(loc=0, scale=5)

plt.legend()

plt.show()
# We can also move the plot of we change the mean of the distribution

plt.figure(figsize=(14, 5))    

plt.subplot(1, 1, 1)    

plot_normal_distribution(loc=0, scale=2)

plot_normal_distribution(loc=3, scale=2)

plot_normal_distribution(loc=6, scale=2)

plt.legend()

plt.show()
def plot_normal_cdf_plot(loc, scale):

    x = np.linspace(-loc-5, loc+5, 10000)

    plt.plot(x, stats.norm.cdf(x, loc=loc, scale=scale), label="stddev=" + str(scale) + ", mean=" + str(loc))

    

plt.figure(figsize=(14, 6))    

plt.subplot(1, 1, 1)     

plot_normal_cdf_plot(loc=0, scale=.5)

plot_normal_cdf_plot(loc=0, scale=1)

plot_normal_cdf_plot(loc=0, scale=1.5)

plot_normal_cdf_plot(loc=0, scale=2.5)

plt.legend()

plt.show()

    