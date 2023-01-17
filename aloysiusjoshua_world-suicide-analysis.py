#import all important library

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from scipy.stats import linregress

warnings.filterwarnings("ignore")
#read the file

data = pd.read_csv("../input/master.csv")

#print the head of the data

print(data.head(3))
#print the preview statistic of each column

print(data.describe())
#total sucide based on the gender

sns.set(style = "whitegrid")

gender_suicide = sns.barplot(x="sex", y="suicides_no", data=data)
#year on year suicide count

yoy_suicide = data.groupby(['year']).sum().reset_index()

#print(yoy_suicide)

plt.figure(figsize=(15,5))

sns.lineplot(x='year', y='suicides_no', data=yoy_suicide, color='navy')

plt.axhline(yoy_suicide['suicides_no'].mean(), ls='--', color='red')

plt.title('Total suicides (by year)')

plt.xlim(1985,2015)

plt.show()
#total suicide by country

suicide_per_country = data.groupby(['country']).mean().sort_values('suicides_no', ascending = False).reset_index()

plt.figure(figsize=(15,20))

sns.barplot(x='suicides_no', y='country', data=suicide_per_country)

plt.axvline(x = suicide_per_country['suicides_no'].mean(), color='red', ls='--')

plt.show()
#total suicide by age group

suicide_per_age = data.groupby(['age']).sum().sort_values('suicides_no', ascending = False).reset_index()

plt.figure(figsize=(10,10))

sns.barplot(x='suicides_no', y='age', data=suicide_per_age)

plt.axvline(x = suicide_per_age['suicides_no'].mean(), color='red', ls='--')

plt.show()
#total suicide by generation

suicide_per_generation = data.groupby(['generation']).sum().sort_values('suicides_no', ascending = False).reset_index()

plt.figure(figsize=(10,10))

sns.barplot(x='suicides_no', y='generation', data=suicide_per_generation)

plt.axvline(x = suicide_per_generation['suicides_no'].mean(), color='red', ls='--')

plt.show()
#year on year suicide of top 3 countries with highest suicide

top3_df = data.loc[(data['country']=='Russian Federation') | (data['country']=='Japan') | (data['country']=='United States') | (data['country']=='France') | (data['country']=='Ukraine') | (data['country']=='Germany') | (data['country']=='Republic of Korea') | (data['country']=='Brazil') | (data['country']=='Poland') | (data['country']=='United Kingdom')].reset_index()

#print(top3_df)

yoy_suicide_top_3 = top3_df.groupby(['year','country']).sum().reset_index()

plt.figure(figsize=(20,10))

sns.lineplot(x='year', y='suicides_no', hue='country', data=yoy_suicide_top_3)

plt.show()
#perason correlation between variable

data.corr(method = 'pearson')
#correlation between suicide_no and population

ds = data.groupby('country')[['population','suicides_no']].corr().iloc[0::2,-1].reset_index()

ds = ds.sort_values('suicides_no', ascending = False)

#print(ds.head())

plt.figure(figsize=(20,20))

sns.barplot(x='suicides_no', y='country', data=ds)

#plt.axvline(x = suicide_per_generation['suicides_no'].mean(), color='red', ls='--')

plt.show()
#year on year suicide count qatar

yoy_suicide_qatar = data.loc[(data['country']=='Qatar')].groupby(['year']).sum().reset_index()

#print(yoy_suicide_qatar)

#print(yoy_suicide)

plt.figure(figsize=(15,8))



plt.subplot(211)

sns.lineplot(x='year', y='suicides_no', data=yoy_suicide_qatar, color='navy')

plt.title('Qatar total suicides (by year)')

plt.xlabel("")



plt.subplot(212)

sns.lineplot(x='year', y='population', data=yoy_suicide_qatar, color='navy')

plt.title('Qatar total population (by year)')



plt.tight_layout

plt.show()