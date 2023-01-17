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

from pathlib import Path

import re

%matplotlib inline





root_dir = '/kaggle/input/novel-corona-virus-2019-dataset/'

filename = root_dir + '2019_nCoV_data.csv'
df = pd.read_csv(filename)



# Checking data shae and example

print("data shape: ", df.shape)

df.head()
# Checking Date Type - Date

df[~ df.Date.map(lambda x : '/' in x)]
# Checking Date Type - Last Update

df[~df['Last Update'].map(lambda x : '/' in x)]
# Dealing Data as Data type (dropping time)



from datetime import date



def to_date(x):

    x = re.split('\/|\-| ', x)

    if len(x[0]) == 4:

        new_date = date(int(x[0]), int(x[1]), int(x[2]))

    else:

        new_date = date(int(x[2]), int(x[0]), int(x[1]))

    return new_date

    

df.Date = df.Date.map(lambda x: to_date(x))

df['Last Update'] = df['Last Update'].map(lambda x: to_date(x))
df.head()
df.tail()
# What is Last Update? -- not sure yet. just leave now.



df[df['Date'] != df['Last Update']]
# sno seems useless -> drop it

df.drop("Sno", axis=1, inplace=True)



#getting a summary of the columns

df.info()
# Summary of numeric variables

df.describe()
# Checking for duplicate rows

duplicate_rows = df.duplicated(['Country', 'Province/State', 'Date'])

df[duplicate_rows]
# List of countries

country_list = list(df.Country.unique())

print(country_list)

print(len(country_list))
# Merging China and mainland China

df.loc[df['Country'] == 'Mainland China', 'Country'] = 'China'
# Viewing by Country

df_bycountry = df.groupby(['Country']).max().reset_index(drop=None)

df_bycountry = df_bycountry.sort_values('Confirmed', ascending=False)



# Drawing Barplot

fig = plt.figure(figsize=(12,6))

g = sns.barplot(x = 'Country', y = 'Confirmed', data = df_bycountry)

g.set_yscale('log')

g.set_xticklabels(g.get_xticklabels(), rotation=45)
# Preparing data for a time-series analysis

df_by_date = df.groupby(['Date']).sum().reset_index(drop=None)

df_by_date['daily_cases'] = df_by_date.Confirmed.diff()

df_by_date['daily_deaths'] = df_by_date.Deaths.diff()

df_by_date['daily_recoveries'] = df_by_date.Recovered.diff()
print(df_by_date)
sns.axes_style("whitegrid")
plt.figure(figsize=(20,5))



sns.barplot(

x='Date',

y='Confirmed',

data=df.groupby(['Date']).sum().reset_index(drop=None))



plt.xticks(rotation=60)

plt.ylabel('Number of confirmed cases', fontsize=15)

plt.xlabel('Dates', fontsize=15)
# plotting two line plots for deaths and recoveries respectively



plt.figure(figsize=(20,10))



plt.plot('Date','Deaths',data=df.groupby(['Date']).sum().reset_index(drop=None),color='red')

plt.plot('Date','Recovered',data=df.groupby(['Date']).sum().reset_index(drop=None),color='blue')



plt.xticks(rotation=60)

plt.ylabel('Number of cases', fontsize=15)

plt.xlabel('Dates', fontsize=15)

plt.legend()

plt.show()
# increasing the figure size

plt.rcParams['figure.figsize'] = (15,7)


