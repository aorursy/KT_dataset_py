import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="darkgrid")

colors = ["amber", "windows blue", "greyish", "faded green", "dusty purple"]

sns.set_palette(sns.xkcd_palette(colors))
sns.set_context("notebook", 1.5)

alpha = 0.7
df0 = pd.read_csv('../input/master.csv')
df0.head()
df0.info()
# Number unique countries

df0['country'].nunique()
#Change columns names

df0.rename(columns={'suicides_no':'suicides', 'suicides/100k pop':'suicides/100k',\

                   ' gdp_for_year ($) ':'gdp/year ', 'gdp_per_capita ($)':'gdp/capita'}, inplace=True)
df0.head()
plt.figure(figsize=(25,13))

# By sex

plt.subplot(221)

sns.countplot(x='sex', data=df0, alpha=alpha, order=['female','male'])

plt.title('Data by sex', fontsize=20)

# By age

plt.subplot(223)

sns.countplot(x='age', data=df0, alpha=alpha)

plt.title('Data by age', fontsize=20)

# By generation

plt.subplot(224)

sns.countplot(x='generation', data=df0, alpha=alpha)

plt.title('Data by generation', fontsize=20)



plt.tight_layout()

plt.show()
plt.figure(figsize=(10,25))

sns.countplot(y='country', data=df0, alpha=alpha)

plt.title('Data by country')

plt.axvline(x=50, color='k')

plt.show()
country_amountData = df0.groupby('country').count()['year'].reset_index()

country_amountData.sort_values(by='year', ascending=True). head(10)
country_selectList = country_amountData[country_amountData['year'] > 50]['country'].reset_index()
df1 = pd.merge(df0, country_selectList, how='outer', indicator=True)

df1 = df1[df1['_merge']=='both']

df1.nunique()
plt.figure(figsize=(25,8))

sns.countplot(x='year', data=df1, alpha=alpha)

plt.title('Data by year')

plt.axhline(y=200, color='k')

plt.show()
df2 = df1[df1['year'] != 2016]
plt.figure(figsize=(10,5))

sns.heatmap(df2.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.show()
number_NAN = len(df2) - df2['HDI for year'].count()

number_noNAN = len(df2)

number_NAN * 100 / number_noNAN
df = df2.drop('HDI for year', axis=1)

df.head()
df.drop(['country-year', 'index', '_merge'], axis=1, inplace=True)
plt.figure(figsize=(10,5))

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.show()
byCountry = df.groupby('country').mean().sort_values('suicides/100k', ascending=False).reset_index()
plt.figure(figsize=(10,25))

sns.barplot(x='suicides/100k', y ='country', data=byCountry)

plt.axvline(x = byCountry['suicides/100k'].mean(),color = 'red', ls='--', linewidth=2)

plt.title('Suicides per 100k (by country)')

plt.show()
byYear = df.groupby('year').mean().reset_index()
plt.figure(figsize=(15,5))

sns.lineplot(x='year', y='suicides/100k', data=byYear, color='navy')

plt.axhline(byYear['suicides/100k'].mean(), ls='--', color='red')

plt.title('Suicides per 100k (by year)')

plt.xlim(1985,2015)

plt.show()
bySex     = df.groupby('sex').mean().reset_index()

bySexYear = df.groupby(['sex','year']).mean().reset_index()

bySexAge = df.groupby(['sex','age']).mean().sort_values('suicides/100k', ascending=True).reset_index()

bySexGeneration = df.groupby(['sex','generation']).mean().sort_values('suicides/100k', ascending=True).reset_index()
plt.figure(figsize=(20,15))

# By sex

plt.subplot(221)

sns.barplot(x='sex', y='suicides/100k', data=bySex, alpha=alpha)

plt.title('Suicides per 100k, by sex')

# Time veolution by sex

plt.subplot(222)

sns.lineplot(x='year', y='suicides/100k', data=bySexYear, hue='sex')

plt.xlim(1985,2015)

plt.title('Evolution suicides per 100k, by sex')

# By sex and age

plt.subplot(223)

sns.barplot(x='sex', y='suicides/100k', data=bySexAge, hue='age', alpha=alpha)

plt.title('Suicides per 100k, by sex and age')

# By sex and generation

plt.subplot(224)

sns.barplot(x='sex', y='suicides/100k', data=bySexGeneration, hue='generation', alpha=alpha)

plt.title('Suicides per 100k, by sex and generation')



plt.tight_layout()

plt.show()
byCountrySex = df.groupby(['country','sex']).mean().reset_index()

byCountrySex.head()

plt.figure(figsize=(10,30))

sns.barplot(y='country', x='suicides/100k', data=byCountrySex, hue='sex')

plt.title('Suicides per 100k (by country and by sex)')

plt.show()
for country in byCountrySex['country']:

    suicides_female = byCountrySex[(byCountrySex['sex']=='female') & \

                                   (byCountrySex['country']==country)]['suicides/100k']

    suicides_male   = byCountrySex[(byCountrySex['sex']=='male')   \

                                   & (byCountrySex['country']==country)]['suicides/100k']

    if suicides_female.iloc[0] > suicides_male.iloc[0]:

        print(country)
byAge     = df.groupby('age').mean().sort_values('suicides/100k', ascending=True).reset_index()

byAgeYear = df.groupby(['age','year']).mean().sort_values('suicides/100k', ascending=True).reset_index()

byAgeSex = df.groupby(['age','sex']).mean().sort_values('suicides/100k', ascending=True).reset_index()

byAgeGen = df.groupby(['age','generation']).mean().sort_values('suicides/100k', ascending=True).reset_index()
plt.figure(figsize=(20,15))

# By age

plt.subplot(221)

sns.barplot(x='age', y='suicides/100k', data=byAge, alpha=alpha)

plt.title('Suicides per 100k, by age')

# Time evolution by age

plt.subplot(222)

sns.lineplot(x='year', y='suicides/100k', data=byAgeYear, hue='age')

plt.xlim(1985,2015)

plt.title('Evolution suicides per 100k, by age')

# 

plt.subplot(223)

sns.barplot(x='age', y='suicides/100k', data=byAgeSex, hue='sex', alpha=alpha)

plt.subplot(224)

sns.barplot(x='age', y='suicides/100k', data=byAgeGen, hue='generation', alpha=alpha)



plt.tight_layout()

plt.show()
g = sns.jointplot(x="gdp/capita", y="suicides/100k", data=byCountry, kind='regresion', \

              xlim=(-100,80000), ylim=(0,45), color='blue')
spain = df0[df0['country']=='Spain']
spain_byYear = spain.groupby('year').mean().reset_index()

spain_bySexYear = spain.groupby(['sex','year']).mean().reset_index()

spain_byAgeYear = spain.groupby(['age','year']).mean().reset_index()
plt.figure(figsize=(20,15))



plt.subplot(211)

sns.lineplot(x='year', y='suicides/100k', data=spain_byYear, color='navy')

plt.axhline(spain_byYear['suicides/100k'].mean(), ls='--', color='red')

plt.axvline(x=2008, ls='-', color='black')

plt.axvline(x=2014, ls='-', color='black')

plt.title('Suicides per 100k in Spain')

plt.xlim(1985,2015)

plt.xlabel('')



plt.subplot(212)

sns.lineplot(x='year', y='gdp/capita', data=spain_byYear, color='navy')

plt.axhline(spain_byYear['gdp/capita'].mean(), ls='--', color='red')

plt.title('GDP in Spain')

plt.xlim(1985,2015)

plt.axvline(x=2008, ls='-', color='black')

plt.axvline(x=2014, ls='-', color='black')



plt.tight_layout

plt.show()
g = sns.jointplot(x="gdp/capita", y="suicides/100k", data=spain_byYear, kind='regresion', color='blue')
plt.figure(figsize=(25,10))

plt.subplot(121)

sns.lineplot(x='year', y='suicides/100k', data=spain_byAgeYear, hue='age')

plt.axvline(x=2008, ls='-', color='black')

plt.axvline(x=2014, ls='-', color='black')

plt.title('Suicides per 100k in Spain, by age')



plt.subplot(122)

sns.lineplot(x='year', y='suicides/100k', data=spain_bySexYear, hue='sex')

plt.xlim(1985,2015)

plt.axvline(x=2008, ls='-', color='black')

plt.axvline(x=2014, ls='-', color='black')

plt.legend(loc='upper right')

plt.title('Suicides per 100k in Spain, by sex')



plt.show()