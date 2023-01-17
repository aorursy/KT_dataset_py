import os # operating system

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # standard graphics

import seaborn as sns # fancier graphics



# Input data files are available in the "../input/" directory.

print(os.listdir("../input"))
# Read the dataset

data = pd.read_csv('../input/avocado.csv')
# Show the first rows

data.head()
data = data.drop(columns = 'Unnamed: 0')

data.head()
data.info()
data['Date'] = pd.to_datetime(data['Date'])

data.info()
data.head()
data = data.rename(columns = {'4046': 'small', '4225': 'large', '4770': 'xl'})

data.head()
# Show the last rows

data.tail()
# Show the descriptive statistics

data.describe()
# What is the mean AveragePrice?

data['AveragePrice'].mean()
# Print it with less decimals

m = data['AveragePrice'].mean()

print('Mean value of AveragePrice = {:.2f}'.format(m))
# What is the standard deviation of Total Volume?

s = data['Total Volume'].std()

s
# Printing the standard deviation of Total Volume in scientific format and with less decimals

print('Standard deviation of Total volume = {:.2e}'.format(s))
# What are the unique type values?

data['type'].unique()
# How many rows there are for each type?

data['type'].value_counts()
# What are the regions?

data['region'].unique()
# How many regions are there in total?

len(data['region'].unique())
data.groupby('type').describe()
# or showing the same values but the table is transposed

data.groupby('type').describe().T
# What if we want only to compare the mean of AveragePrice between different types?

data['AveragePrice'].groupby(data['type']).mean()
# Another example: How does the sum of large avocados vary grouped by year?

data['large'].groupby(data['year']).sum()
# What is the distribution of the average prices?

data.hist(column = 'AveragePrice', bins = 30, figsize = (8, 6))

plt.xlabel('Price')

plt.ylabel('Count')

plt.title('Distribution of avocado average prices')

plt.show()
# How do the prices differ by type?

ax = data.hist(column = 'AveragePrice', by = 'type', bins = 30, sharex = True, grid = True, figsize = (14, 6), xlabelsize = 12, ylabelsize = 12)

# Annotate the graphs (xlabel, ylabel and grid)

for i in range(2):

    ax[i].set_xlabel('Price', size = 14)

    ax[i].set_ylabel('Count', size = 14)

    ax[i].grid()

plt.show()
# Use seaborn to create distribution plot

import warnings



# sns.distplot() gives some warnings. Ignore them.

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    plt.figure(figsize=(12,5))

    plt.title("Distribution of avocado average price")

    ax = sns.distplot(data["AveragePrice"], color = 'r')

    plt.grid()
# Or quick help, remove the comment

# ?sns.distplot
# Can we overlay the distribution of average price grouped by type?



# sns.distplot() gives some warnings. Ignore them.

with warnings.catch_warnings():

    warnings.simplefilter("ignore")



    # Create a figure, add title

    plt.figure(figsize=(12,5))

    plt.title("Distribution of average price grouped by type")



    # Plot the distribution of conventional type data

    mask0 = data['type'] == 'conventional'

    ax = sns.distplot(data["AveragePrice"][mask0], color = 'r', label = 'conventional')



    # Plot the histogram of organic type data

    mask1 = data['type'] == 'organic'

    ax = sns.distplot(data["AveragePrice"][mask1], color = 'g', label = 'organic')



    # add legend, show the graphics

    plt.legend()

    plt.grid()
# Make a boxplot graph using pandas

data.boxplot(column = 'AveragePrice', by = 'type', figsize = (8,6))

plt.show()
# Make a boxplot graph with seaborn

plt.figure(figsize=(12,5))

sns.boxplot(y = "type", x = "AveragePrice", data = data, palette = 'pink')

plt.xlim([0, 4])

plt.grid()

plt.show()
# Violin plot using seaborn



# sns.violinplot() gives some warnings. Ignore them.

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    plt.figure(figsize=(12,5))

    sns.violinplot(y = "type", x = "AveragePrice", data = data, palette = 'pink')

    plt.xlim([0, 4])

    plt.grid()

    plt.show()
# How do the avocado prices vary by region? First type == 'organic'

mask = data['type'] == 'organic'

g = sns.catplot(x = 'AveragePrice', y = 'region', data = data[mask],

                   height = 13,

                   aspect = 0.8,

                   palette = 'magma')

plt.xlim([0, 4])

plt.grid()

plt.show()
# How about conventional avocados? Their price distribution by region?

mask = data['type'] == 'conventional'

g = sns.catplot(x = 'AveragePrice', y = 'region', data = data[mask],

                   height = 13,

                   aspect = 0.8,

                   palette = 'magma')

plt.xlim([0, 4])

plt.grid()

plt.show()
# Overlay the distributions, we use hue = 'type' for overalying

g = sns.catplot(data = data,

                x = 'AveragePrice', 

                y = 'region',

                hue = 'type',

                height = 13,

                aspect = 0.8,

                palette = 'magma')

plt.xlim([0, 4])

plt.grid()

plt.show()