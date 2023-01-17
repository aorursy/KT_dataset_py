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
# Let's take some distribution and see if CLT really works or not

import scipy.stats as stats



def plot_sample_means(sample_size, number_of_samples, dist):

    rvs_dist = []

    if dist == "expon":

        # Generate 1000 points from an exponential distribution

        rvs_dist = stats.expon().rvs(10000)

    elif dist == "uniform":

        # Generate 1000 points from an exponential distribution

        rvs_dist = stats.uniform().rvs(10000)        

    # Taking 100 samples of size 30    

    sample_means = [np.random.choice(a=rvs_dist, size=sample_size, replace=True).mean() for _ in range(0, number_of_samples)]

    sns.distplot(sample_means, label="number of different samples: " + str(number_of_samples))
plot_sample_means(sample_size=30, number_of_samples=300, dist="expon")
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)

plot_sample_means(sample_size=30, number_of_samples=50, dist="expon")

plt.subplot(2, 2, 2)

plot_sample_means(sample_size=30, number_of_samples=200, dist="expon")

plt.subplot(2, 2, 3)

plot_sample_means(sample_size=30, number_of_samples=800, dist="expon")

plt.subplot(2, 2, 4)

plot_sample_means(sample_size=30, number_of_samples=1500, dist="expon")
# Let' change the size of each sample

plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)

plot_sample_means(sample_size=30, number_of_samples=200, dist="expon")

plt.subplot(2, 2, 2)

plot_sample_means(sample_size=120, number_of_samples=200, dist="expon")

plt.subplot(2, 2, 3)

plot_sample_means(sample_size=300, number_of_samples=200, dist="expon")

plt.subplot(2, 2, 4)

plot_sample_means(sample_size=1000, number_of_samples=10000, dist="expon")
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)

plot_sample_means(sample_size=30, number_of_samples=200, dist="uniform")

plt.subplot(2, 2, 2)

plot_sample_means(sample_size=120, number_of_samples=200, dist="uniform")

plt.subplot(2, 2, 3)

plot_sample_means(sample_size=300, number_of_samples=200, dist="uniform")

plt.subplot(2, 2, 4)

plot_sample_means(sample_size=1000, number_of_samples=10000, dist="uniform")