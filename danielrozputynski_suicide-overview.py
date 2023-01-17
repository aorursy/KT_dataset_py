import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



df = pd.read_csv('../input/master.csv')
#Lets see how the date looks like

df.head()
df.shape


df['country'].unique()


df['country'].nunique()
df.dtypes
#I may use the gdp_for_year  so i want to convert this column in floats

df[' gdp_for_year ($) '] = df[' gdp_for_year ($) '].apply(lambda x: x.replace(',','')).astype(float)
df.dtypes
#I would like to know if there is some NaN values

df.isnull().any()
df['HDI for year'].isna().sum()
fig = plt.figure(figsize=(20,2))

sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'ocean')
#As we see there is a plenity of them so it is a little bit useless in our analyse, Maybe later we are gonna use that to see in which countries there is more NaN values

df = df.drop(columns = 'HDI for year')
plt.subplots(figsize=(15,10))

sns.heatmap(df.corr(), annot=True, linewidths = 0.3)

plt.title('Correlation of the dataset', size=16)

plt.show()
#Let's see how the suicides per gender are distributed

dfSex=df.groupby(["sex"])["suicides_no"].sum().reset_index()

sns.barplot(x="sex", y="suicides_no", data=dfSex, palette="Blues_d")

plt.show()
#Let's see how the suicides per age are distributed

dfAge = df.groupby(['age'])['suicides_no'].sum().reset_index()

dfAge = dfAge.sort_values(by='suicides_no',ascending=False)

plt.subplots(figsize=(10,6))

sns.barplot(x='age', y='suicides_no', data=dfAge, palette = 'Blues_d')
#Let's see how the suicides per generation are distributed

dfGeneration = df.groupby(['generation'])['suicides_no'].sum().reset_index()

dfGeneration = dfGeneration.sort_values(by='suicides_no',ascending=False)

plt.subplots(figsize=(10,6))

sns.barplot(x='generation', y='suicides_no', data=dfGeneration, palette = 'Blues_d')
dfCountry = df.groupby(['country'])['suicides_no'].sum().reset_index()

dfCountry = dfCountry.sort_values('suicides_no',ascending=False)

dfCountry = dfCountry.head(10)



plt.subplots(figsize=(15,6))

sns.barplot(x='country', y='suicides_no', data=dfCountry, palette = 'Blues_d')
array = ['Russian Federation', 'United States', 'Japan', 'France', 'Ukraine', 'Germany', 'Republic of Korea', 'Brazil', 'Poland', 'United Kingdom']

dfPeriod = df.loc[df['country'].isin(array)]

dfPeriod = dfPeriod.groupby(['country', 'year'])['suicides_no'].sum().unstack('country').plot(figsize=(20, 7))

dfPeriod.set_title('Top suicide countries', size=15, fontweight='bold')
dfSexPeriod =df.groupby(['sex', 'year'])['suicides_no'].sum().unstack('sex').plot(figsize=(20, 7))

dfSexPeriod.set_title('Suicide per Sex', size=15, fontweight='bold')

dfAgePeriod =df.groupby(['age', 'year'])['suicides_no'].sum().unstack('age').plot(figsize=(20, 10))

dfAgePeriod.set_title('Suicide per Age', size=15, fontweight='bold')