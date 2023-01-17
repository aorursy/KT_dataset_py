# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#I like to add silly comments to see if people read them



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline  





# read that data in!

train = pd.read_hdf('../input/train.h5')
print(train.shape)

train.head(10)
#well well well, my old friend NaN is here.  I know how to deal with him though.

#lets make a bunch of histograms and look at how my data is distributed

colnames=list(train.columns.values)

#since I know what is about to happen, I'll save you 90 graphs of repeats

colnames=colnames[1:5]

for i in colnames:

    data=train.loc[train.loc[:,i].notnull(),i]

    #data=data.loc[(data>np.percentile(data,1)) & (data<np.percentile(data,99))]

    sns.distplot(data,kde=False,hist=True)

    plt.show()
#that did not look good.  lets take a look at a cdf for one of those "spikes"

sorted_data = np.sort(train['derived_0'])  

plt.step(sorted_data, np.arange(sorted_data.size)/sorted_data.size) 

plt.title('derived_0 CDF')

plt.show()
#hmm it looks like we need to zoom in and will see that near zero we have a normal variable

#this means we have large valued outliers.

sorted_data = np.sort(train['derived_0'])  

plt.step(sorted_data, np.arange(sorted_data.size)/sorted_data.size) 

plt.title('derived_0 CDF')

plt.xlim(-1,1)

plt.show()
#Well that was not cool.  It looks like we have outliers.  

#since we are just looking, lets kill those outliers and do it dynamically.

#use np.percentile(x,n) to get six-sigma worth of data

colnames=list(train.columns.values)

for i in colnames:

    data=train.loc[train.loc[:,i].notnull(),i]

    data=data.loc[(data>np.percentile(data,0.5)) & (data<np.percentile(data,99.5))]

    sns.distplot(data,kde=False,hist=True)

    plt.show()