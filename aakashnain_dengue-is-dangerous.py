# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

color = sns.color_palette()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/denguecases.csv')
data.head()
# A quick overview of the dataset

data.info()
# Check for any null/missing values

data.isnull().sum()
# The number of cases in any month corresponding to a year are given as floats. Let's round off

data['Dengue_Cases'] = data['Dengue_Cases'].apply(lambda x : np.round(x).astype(np.uint8))

data.head()
# How many regions are there?

regions_count = data.Region.value_counts()

print("Total number of regions : ", len(regions_count))

print("Regions along with the number of occurrences")

print(regions_count)
print("Minimum number of dengue cases : ", np.min(data.Dengue_Cases.values))

print("Maximum number of dengue cases : ", np.max(data.Dengue_Cases.values))

print("Average number of dengue cases : ", int(np.mean(data.Dengue_Cases.values)))
# What's the count of cases for each year?

data_groups = data.groupby(['Year'])

year = []

cases = []



for name, group in data_groups:

    year.append(name)

    cases.append(group['Dengue_Cases'].sum())



plt.figure(figsize=(10,8))    

sns.barplot(x=year, y=cases, color=color[2])

plt.title('Total number of dengue cases in an year')

plt.xlabel('Year', fontsize=16)

plt.ylabel('Count', fontsize=16)

plt.show()
# What about the regions? Waht are the cases count as per the regions?

data_groups = data.groupby(['Region'])



regions = []

cases = []



for name, group in data_groups:

    regions.append(name)

    cases.append(group['Dengue_Cases'].sum())



plt.figure(figsize=(10,8))    

sns.barplot(y=regions, x=cases, color=color[3], orient='h')

plt.title('Region-wise number of dengue cases')

plt.xlabel('Count', fontsize=16)

plt.ylabel('Region', fontsize=16)

plt.yticks(range(len(regions)), regions)

plt.show()
# What about the count per region in each year?

data_groups = data.groupby(['Year'])



f,axs = plt.subplots(5,2, figsize=(20,40), sharex=False, sharey=False)

for i,(year, group) in enumerate(data_groups):

    regions = []

    cases = []

    region_group = group.groupby(['Region'])

    for region, df in region_group:

        regions.append(region)

        cases.append(df['Dengue_Cases'].sum())

    sns.barplot(y=regions, x=cases, color=color[4], orient='h', ax=axs[i//2, i%2])

    axs[i//2, i%2].set_title(year, fontsize=14)

    axs[i//2, i%2].set_xlabel("Cases", fontsize=14)

    axs[i//2, i%2].set_ylabel("Region", fontsize=14)

f.delaxes(axs[4][1])

plt.show()    
# Let's do some monthly analysis now.

data_groups = data.groupby(['Year'])

months = ('Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

f,axs = plt.subplots(5,2, figsize=(20,40), sharex=False, sharey=False)

for i,(year, group) in enumerate(data_groups):

    sns.barplot(y=group['Month'], x=group['Dengue_Cases'], color=color[5], orient='h', ax=axs[i//2, i%2])

    axs[i//2, i%2].set_title(year, fontsize=14)

    axs[i//2, i%2].set_xlabel("Cases", fontsize=14)

    axs[i//2, i%2].set_ylabel("Month", fontsize=14)

    axs[i//2, i%2].set_xticks(range(len(months)), months)

f.delaxes(axs[4][1])

plt.show()    