# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_levels= pd.read_csv('../input/chennai-water-management/chennai_reservoir_levels.csv')

df_levels.head()
def ecdf(data):

    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n

    #____ = ____(____)

    n= len(data)

    # x-data for the ECDF: x

    #____ = ____(____)

    x=np.sort(data)

    # y-data for the ECDF: y

    #____ = ____(____, ____) / n

    y=np.arange(start=1,stop=n+1)/n

    return x, y

def bootstrap_replicate_1d(data, func):

    return func(np.random.choice(data, size=len(data)))
def draw_bs_reps(data, func, size=1):

    """Draw bootstrap replicates."""

    

    # Initialize array of replicates: bs_replicates

    bs_replicates = np.empty(size)



    # Generate replicates

    for i in range(size):

        bs_replicates[i] = bootstrap_replicate_1d(data,func)



    return bs_replicates

def confidence_interval(data,arr):

    conf_int = np.percentile(data,arr)

    return conf_int 
# Take 10,000 bootstrap replicates of the mean: bs_replicates

bs_replicates = draw_bs_reps(df_levels['POONDI'],np.mean,10000)



# Compute and print SEM

sem = np.std(df_levels['POONDI']) / np.sqrt(len(df_levels))

print('normalized mean',sem)



# Compute and print standard deviation of bootstrap replicates

bs_std = np.std(bs_replicates)

print('bootstrap mean',bs_std)



#Confidence interval

print('confidence interval',confidence_interval(bs_replicates,[2.5,97.5]))





# Make a histogram of the results

_ = plt.hist(bs_replicates, bins=50, normed=True)

_ = plt.xlabel('mean annual water level at poondi (mm)')

_ = plt.ylabel('PDF')



# Show the plot

plt.show()
# Take 10,000 bootstrap replicates of the mean: bs_replicates

bs_replicates = draw_bs_reps(df_levels['CHOLAVARAM'],np.mean,10000)



# Compute and print SEM

sem = np.std(df_levels['CHOLAVARAM']) / np.sqrt(len(df_levels))

print('normalized mean',sem)



# Compute and print standard deviation of bootstrap replicates

bs_std = np.std(bs_replicates)

print('bootstrap mean',bs_std)



#Confidence interval

print('confidence interval',confidence_interval(bs_replicates,[2.5,97.5]))





# Make a histogram of the results

_ = plt.hist(bs_replicates, bins=50, normed=True)

_ = plt.xlabel('mean annual water level at cholavaram (mm)')

_ = plt.ylabel('PDF')



# Show the plot

plt.show()
# Take 10,000 bootstrap replicates of the mean: bs_replicates

bs_replicates = draw_bs_reps(df_levels['REDHILLS'],np.mean,10000)



# Compute and print SEM

sem = np.std(df_levels['REDHILLS']) / np.sqrt(len(df_levels))

print('normalized mean',sem)



# Compute and print standard deviation of bootstrap replicates

bs_std = np.std(bs_replicates)

print('bootstrap mean',bs_std)



#Confidence interval

print('confidence interval',confidence_interval(bs_replicates,[2.5,97.5]))





# Make a histogram of the results

_ = plt.hist(bs_replicates, bins=50, normed=True)

_ = plt.xlabel('mean annual water level at in redhills (mm)')

_ = plt.ylabel('PDF')



# Show the plot

plt.show()
# Take 10,000 bootstrap replicates of the mean: bs_replicates

bs_replicates = draw_bs_reps(df_levels['CHEMBARAMBAKKAM'],np.mean,10000)



# Compute and print SEM

sem = np.std(df_levels['CHEMBARAMBAKKAM']) / np.sqrt(len(df_levels))

print('normalized mean',sem)



# Compute and print standard deviation of bootstrap replicates

bs_std = np.std(bs_replicates)

print('bootstrap mean',bs_std)



#Confidence interval

print('confidence interval',confidence_interval(bs_replicates,[2.5,97.5]))





# Make a histogram of the results

_ = plt.hist(bs_replicates, bins=50, normed=True)

_ = plt.xlabel('mean annual water level at in chembarambakkam')

_ = plt.ylabel('PDF')



# Show the plot

plt.show()