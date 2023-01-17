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
s=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
s.head()
s.tail()
s.shape



s.isnull().sum()



del s['HDI for year']
s_max=s.groupby(by='country')['suicides_no'].sum().reset_index().sort_values(by='suicides_no',ascending=False)
s_max.head()
sns.barplot(x='country',y='suicides_no',data=s_max.head())

plt.xticks(rotation=90)

plt.show()
s_min=s.groupby(by='country')['suicides_no'].sum().reset_index().sort_values(by='suicides_no')
s_min.head()
sns.barplot(x='country',y='suicides_no',data=s_min.head())

plt.xticks(rotation=90)

plt.show()
s_danger=s.loc[s.country.isin(s_max.country.head())]

s_danger.country.unique()
sns.boxplot(x='age',y='suicides_no',data=s_danger)

plt.xticks(rotation=90)

plt.show()

#It shows the suicide distribution for various age groups in 5 danger countries

#where suicides are maximum in age group 35-54 years and 5-14yrs age group having least no of suicides
s_safe=s.loc[s.country.isin(s_min.country.head())]

s_safe.country.unique()
s_safe.groupby('age')['suicides_no'].mean().plot.pie(autopct="%1.1f%%",explode=(0.5,0.5,0.5,0.5,0.5,0.5))

#It shows the suicide mean percentage for various age groups in 5 safe countries

sns.barplot(x='country',y='gdp_per_capita ($)',data=s_danger)

plt.xticks(rotation=45)

plt.show()
sns.barplot(x='country',y='gdp_per_capita ($)',data=s_safe)

plt.xticks(rotation=45)

plt.show()
fig=plt.figure(figsize=(15,5))

sns.swarmplot(x='generation',y='suicides_no',data=s_danger,hue='country')

plt.show()
fig=plt.figure(figsize=(15,5))

sns.violinplot(x='generation',y='suicides_no',data=s_safe)

plt.show()
sns.barplot(x='sex',y='suicides_no',hue='age',data=s_danger)

plt.show()
sns.barplot(x='sex',y='suicides_no',hue='age',data=s_safe)

plt.show()