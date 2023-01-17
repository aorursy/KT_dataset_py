# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # graphing

from matplotlib.ticker import MaxNLocator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Read data

df = pd.read_csv('/kaggle/input/covidnet-hospitalization-rates/COVID-NET_Surveillance_03-28-2020.csv')
# Sample data

df.head()
# Describe data

df.describe()
# Remove future weeks (with values set to Nan)

df = df.dropna()
df.describe()
# What are the catchments (regions)?

catchments = df.CATCHMENT.unique()
catchments
# What are the networks?

df.NETWORK.unique()
# What are the age categories?

ages = df.AGE_CATEGORY.unique()
ages
# Show trend by age

# Remove a few categories for clarity in graph
ages = ages[ages != '65+ yr']
ages = ages[ages != '85+']
ages = ages[ages != 'Overall']
df_ages = df[df.AGE_CATEGORY.isin(ages)]

df_ages = df_ages[df_ages.CATCHMENT == 'Entire Network']

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
sns.lineplot(x=df_ages['MMWR-WEEK'], y=df_ages.CUMULATIVE_RATE, hue=df_ages.AGE_CATEGORY, ax=ax1)
sns.lineplot(x=df_ages['MMWR-WEEK'], y=df_ages.WEEKLY_RATE, hue=df_ages.AGE_CATEGORY, ax=ax2)
fig.show()

# Show trend by catchment (region)

df_catchments = df[df.AGE_CATEGORY == 'Overall']

# Remove 'Entire Network' value for clarity in graph
catchments = catchments[catchments != 'Entire Network']
df_catchments = df_catchments[df_catchments.CATCHMENT.isin(catchments)]

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
sns.lineplot(x=df_catchments['MMWR-WEEK'], y=df_catchments.CUMULATIVE_RATE, hue=df_catchments.CATCHMENT, ax=ax1)
sns.lineplot(x=df_catchments['MMWR-WEEK'], y=df_catchments.WEEKLY_RATE, hue=df_catchments.CATCHMENT, ax=ax2)
fig.show()