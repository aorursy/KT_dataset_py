import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('../input/kaggle_datasets.csv')

data.head()
data.shape
data_ex0 = data.loc[(data.kernels > 0 ) | (data.upvotes > 0), :]

data_ex0.shape
data_ex0.describe()
plt.hist(data_ex0.discussions); # The ';' is to avoid showing a message before the chart
data_ex0.discussions.plot(kind='hist'); 
data_ex0.nlargest(10, 'discussions').loc[:,['title','discussions']]
# Default plot

plt.figure(figsize=(8,5)) # Specify the figure size

plt.hist(data_ex0.kernels)

plt.show()
# Zoom in to distribution of 0-20 kernels

plt.figure(figsize=(8,5))

plt.hist(data_ex0.kernels, range = (0, 21))

plt.show()
# Look at the tail ends

plt.figure(figsize=(8,5))

plt.hist(data_ex0.kernels, range = (100, data_ex0.kernels.max())) # 100 up to highest number of kernels

plt.show()
# Zero to 100 kernels in 20 bins

plt.figure(figsize=(8,5))

plt.hist(data_ex0.kernels, range = (0, 100), bins = 20)

plt.show()
# Use a numpy array to specify how the bins are separated

plt.figure(figsize=(8,5))

plt.hist(data_ex0.kernels, bins = np.arange(5, 51, 5)) # 5-10, 10-15... up to 45-50

plt.show()
# Taking logarithm on the x-axis

plt.figure(figsize=(8,5))

plt.hist(np.log1p(data_ex0.kernels)) # Use np.log1p instead of np.log to avoid error taking log of 0

plt.show()
# Taking logarithm on the y-axis

plt.figure(figsize=(8,5))

plt.hist(data_ex0.kernels, bins=30, log=True)

plt.show()
plt.figure(figsize=(8,5))

plt.hist(data_ex0.kernels, bins=30, log=True, color = 'fuchsia')

plt.title('Distribution of Kernels Created', fontsize=16)

plt.xlabel('No. of kernels')

plt.ylabel('Frequency')

plt.show()
data_ex0.upvotes.plot.kde();
data_ex0.upvotes.plot.kde(ind = np.arange(0, data_ex0.upvotes.max()));
sns.kdeplot(data_ex0.upvotes);
sns.kdeplot(data_ex0.upvotes, clip = (0,200));
# Do it with Seaborn

plt.figure(figsize=(8,5))

sns.kdeplot(data_ex0.loc[data_ex0.featured == 0, 'upvotes'], color='green', label='non-featured')

sns.kdeplot(data_ex0.loc[data_ex0.featured == 1, 'upvotes'], color='red', label='featured')

plt.xlim(0, 100) # Limit the view from 0 to 100

plt.show()
# Do it with Matplotlib

plt.figure(figsize=(8,5))

data_ex0.loc[data_ex0.featured == 1, 'upvotes'].plot.kde(color='red')

data_ex0.loc[data_ex0.featured == 0, 'upvotes'].plot.kde(color='green')

plt.legend(('Yes', 'No'), title='Featured?')

plt.xlim(0,100)

plt.show()