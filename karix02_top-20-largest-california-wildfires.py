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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set(style="darkgrid")
df_ = pd.read_csv("/kaggle/input/top-20-largest-california-wildfires/top_20_CA_wildfires.csv")

df = df_.copy()
df.head()
df.tail()
df.info()
df.describe()
df['month'].unique()
df['month'].nunique()
df['cause'].unique()
'''

Helper functions

'''

def create_df(data, columns):

    return pd.DataFrame(data=data, columns=columns)



def barplot(x, y, data, columns):

    plt.figure(figsize=(15, 7))

    return sns.barplot(x=x, y=y, data=create_df(data, columns))



def lineplot(x, y, data, columns):

    plt.figure(figsize=(15, 7))

    return sns.lineplot(x=x, y=y, data=create_df(data, columns))
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',

          'August', 'September', 'October', 'November', 'December']

per_month = np.zeros((len(months)))



for i, month in enumerate(months):

    per_month[i] = df.loc[df['month'] == month].shape[0]
barplot(x='Month', y='Count', data=np.c_[months, per_month], columns=['Month', 'Count'])
years = df['year'].unique() # get all unique years

'''

I will use bubble sort because it is easey to implement and in our small amount of data O(n^2) is not big

'''

for i in range(years.shape[0]):

    for j in range(years.shape[0] - 1):

        if years[j] > years[j+1]:

            years[j], years[j+1] = years[j+1], years[j]
per_year = np.zeros(years.shape)



for i, year in enumerate(years):

    per_year[i] = df.loc[df['year'] == year].shape[0]
barplot(x='Year', y='Count', data=np.c_[years.astype(str), per_year], columns=['Year', 'Count'])
total_per_year = np.zeros(years.shape)

total_per_year[0] = per_year[0]



for i in range(1, per_year.shape[0]):

    total_per_year[i] = total_per_year[i - 1] + per_year[i]
df['year'].min(), df['year'].max()
lineplot(x='Year', y='Total count', data=np.c_[years, total_per_year], columns=['Year', 'Total count']).set(xticks=np.arange(1930, 2030, 10))
causes = df['cause'].unique()

deaths_per_cause = np.zeros(causes.shape)



for i, cause in enumerate(causes):

    deaths_per_cause[i] = df[df['cause'] == cause]['deaths'].values.sum() * 100 / df['deaths'].values.sum()
barplot(x='Cause', y='Deaths', data=np.c_[causes, deaths_per_cause], columns=['Cause', 'Deaths']).set(ylabel='Deaths [%]')
acres_burned_by_cause = np.zeros(causes.shape)



for i, cause in enumerate(causes):

    acres_burned_by_cause[i] = df.loc[df['cause'] == cause]['acres'].values.sum()
barplot(x='Cause', y='Acres', data=np.c_[causes, acres_burned_by_cause], columns=['Cause', 'Acres'])
acres_per_year = np.zeros(years.shape)



for i, year in enumerate(years):

    acres_per_year[i] = df.loc[df['year'] == year]['acres'].values.sum()
barplot(x='Year', y='Acres burned', data=np.c_[years.astype(str), acres_per_year], columns=['Year', 'Acres burned'])
total_acres = np.zeros(years.shape)

total_acres[0] = acres_per_year[0]



for i in range(1, acres_per_year.shape[0]):

    total_acres[i] = total_acres[i-1] + acres_per_year[i]
lineplot(x='Year', y='Total acres burned', data=np.c_[years, total_acres], columns=['Year', 'Total acres burned']).set(xticks=np.arange(1930, 2030, 10))