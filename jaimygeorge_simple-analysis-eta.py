

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

plt.rc('figure', figsize = (15, 10))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading the dataset file

df = pd.read_csv("../input/master.csv")
# Displaying the first 5 rows of the DataFrame

df.head()
# Data type in each column

df.dtypes
# Summary of information in all columns

df.describe().round(2)
# Number of rows and columns in the DataFrame

print('Number of rows:', df.shape[0])

print('Number of columns:', df.shape[1])
# Verify the country that the highest GDP per Capita

df[df['gdp_per_capita ($)'] == 126352.000000]
print("Luxembourg has the highest GDP per capita")
df_luxembourg = df.query("country == 'Luxembourg'")
df_luxembourg.head(10)
year_sum_luxembourg = df_luxembourg.groupby('year').sum()['suicides_no'].sort_values(ascending=False).reset_index()

figure = sns.barplot(x = 'year', y = 'suicides_no', data = year_sum_luxembourg, palette="BuGn_r", order=year_sum_luxembourg['year'])

figure.set_title('Number of suicides per year in Luxembourg', {'fontsize': 22})

figure.set_xlabel('Year', {'fontsize': 18})

figure.set_ylabel('Total', {'fontsize': 18})

plt.rcParams["xtick.labelsize"] = 10

plt.xticks(rotation= 90)
new_df = pd.DataFrame(df.groupby('country').sum()['suicides_no'].sort_values(ascending=False).reset_index())

analysing_total = new_df.head(10)
figure = sns.barplot(x = 'country', y = 'suicides_no', data = analysing_total, palette="GnBu_d")

figure.set_title('Total of the suicides between 1985-2016', {'fontsize': 22})

figure.set_xlabel('Country', {'fontsize': 18})

figure.set_ylabel('Total', {'fontsize': 18})

plt.rcParams["xtick.labelsize"] = 3

plt.xticks(rotation= 90)
countries_oceania = ['New Zealand', 'Australia']

df_ne = df[df['country'].isin(countries_oceania)]

ax = df_ne.groupby(['country', 'year'])['suicides/100k pop'].sum().unstack('country').plot(figsize=(10, 10))

ax.set_title('Suicides in Oceania', fontsize=20)

ax.legend(fontsize=15)

ax.set_xlabel('Year', fontsize=20)

ax.set_ylabel('Suicides Number', fontsize=20)

ax
countries_oceania = ['New Zealand', 'Australia']

for country in countries_oceania:

    grouped = df[df['country'] == country].groupby(['year', 'age'])['suicides/100k pop'].sum().unstack('age')

    grouped.plot(figsize=(10, 10),

               title='Suicides per 100k population by age in ' + country,

               legend=True)
new_zealand_analysis = df[df['country'] == 'New Zealand'].groupby(['year', 'age'])['suicides/100k pop'].sum().unstack('age')
new_zealand_analysis.plot(figsize=(10, 10),

               title='Suicides per 100k population by age in New Zealand',

               legend=True)