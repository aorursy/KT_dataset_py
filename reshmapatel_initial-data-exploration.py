
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
train_csv=pd.read_csv("../input/train.csv")
train_csv.describe()
train_csv.columns
train_csv['target'].describe()
target_var=train_csv['target']
target_var.count()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
plt.scatter(range(train_csv.shape[0]), np.sort(train_csv['target'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Target Distribution", fontsize=14)
plt.show()
#Generate Kernel Density Estimate plot using Gaussian kernels.

# In statistics, `kernel density estimation`_ (KDE) is a non-parametric
# way to estimate the probability density function (PDF) of a random
# variable. This function uses Gaussian kernels and includes automatic
# bandwith determination.
# A density plot is a smoothed, continuous version of a histogram estimated from 
# the data. The most common form of estimation is known as kernel density estimation-kde. 
# In this method, a continuous curve (the kernel) is drawn at every individual data 
# point and all of these curves are then added together to make a single smooth density 
# estimation. The kernel most often used is a Gaussian (which produces a Gaussian bell 
# curve at each data point).

target_var.plot.kde()
# histogram
target_var.plot.hist()
sns.distplot(target_var, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# hence the target data is right skewed distribution 
# logarithmic distribution gives the better distribution of frequency
target_log=np.log1p(target_var)

plt.figure(figsize=(12,8))
#target_log.plot.hist(bins=50)

plt.hist(x=target_log,bins=50,color='darkred')
plt.title('Logarithmic Distribution')
plt.xlabel('Target')
#plt.c
plt.show()
#plt.grid(True)
# bootstrap plot for actual data shows uncertainty of statistics

#The bootstrap plot is used to estimate the uncertainty of a statistic by 
#relaying on random sampling with replacement [R33]. This function will generate 
#bootstrapping plots for mean, median and mid-range statistics 
#for the given number of samples of the given size.

from pandas.plotting import bootstrap_plot
fg=plt.figure(figsize=(12,8))

bootstrap_plot(target_var, size=50, samples=500, color='lightblue',fig=fg)
plt.show()


