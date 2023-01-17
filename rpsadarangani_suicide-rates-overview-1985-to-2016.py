# Import all the needed python libary for data visualtion and Analysis

import pandas as pd

import numpy as np

import matplotlib

from matplotlib import pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/master.csv')
# Check the Data Infomation

data.info()
print(data.columns)

data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicides/100k pop', 'country-year', 'HDI for year',

       'gdp_for_year', 'gdp_per_capita', 'generation']
data.corr()
# Correlation Map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)

plt.show()
# Check the first 10 Record

data.head(10)
# Check the last 10 Record

data.tail(10)
# Check all Country in the dataset

data.country.unique()
data.country.nunique()
data.describe()
data.describe(include='object')
data.isnull().sum()
data['age'].unique()
# Before we explore the data let us fill the HDI with 0

data.fillna(0, axis=1, inplace=True)
data.isnull().sum()
grouped = data.groupby(['country',])['suicides_no'].sum().reset_index()

grouped = grouped.sort_values('suicides_no', ascending=False)

grouped = grouped[:10]
f,ax = plt.subplots(figsize=(20, 10))

sns.barplot(x='country', y='suicides_no', data=grouped)

plt.title('Top Country for Suicide 1985 to 2016')

plt.xlabel('Country')

plt.ylabel('Number of Suicide')
grouped = data.groupby(['country',])['suicides_no', 'population'].sum().reset_index()

grouped['Suicide / Population'] = grouped['suicides_no'] / grouped['population']

grouped = grouped.sort_values('Suicide / Population', ascending=False)

grouped = grouped[:10]

grouped
f,ax = plt.subplots(figsize=(20, 10))

sns.barplot(x='country', y='Suicide / Population', data=grouped)

plt.title('Top Country for Suicide based on Population 1985 to 2016')

plt.xlabel('Country')

plt.ylabel('Sucide / Population')
grouped = data.groupby('year')['suicides_no'].sum().reset_index()

grouped = grouped.sort_values('suicides_no', ascending=False).reset_index(drop=True)

grouped = grouped[:10]

grouped
f,ax = plt.subplots(figsize=(20, 10))

sns.barplot(x='year', y='suicides_no', data=grouped)

plt.title('Top Year for Suicide')

plt.xlabel('Year')

plt.ylabel('Number of Suicide')
grouped = data.groupby(['age'])['suicides_no'].sum().reset_index()

grouped = grouped.sort_values('suicides_no', ascending=False).reset_index(drop=True)

grouped
f,ax = plt.subplots(figsize=(20, 10))

sns.barplot(x='age', y='suicides_no', data=grouped)

plt.title('Suicide According Age Group')

plt.xlabel('Age Group')

plt.ylabel('Number of Suicide')
fig, ax = plt.subplots(figsize=(15,7))

grouped = data.groupby(['year', 'sex'])['suicides_no'].sum().unstack().plot(ax=ax)

plt.xlabel('Year')

plt.ylabel('Number of Suicide')

plt.title('Genderwise Suicide Comparison')
data.corr()['suicides_no']
data.corr()['population']