import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
data = pd.read_csv('../input/master.csv')

data.head()
data.columns.values
data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicidesper100kpop', 'country-year', 'HDI for year',

       'gdp_for_year_dollars', 'gdp_per_capita_dollars', 'generation']

data.columns.values
data['gdp_for_year_dollars'] = data['gdp_for_year_dollars'].str.replace(',','').astype(int)
data.info()
data.isnull().sum().sort_values(ascending=False)
data_n = data.drop(['HDI for year', 'country-year'], axis=1)

data_n.head(3)
data_n.describe()
data_n.describe(include=['O'])
data_n[['sex','suicides_no']].groupby(['sex']).mean().sort_values(by='suicides_no', ascending=False).plot(kind='bar')
plt.figure(figsize=(10,5))

sns.barplot(x = 'age', y='suicides_no', hue='sex', data=data_n.groupby(["age","sex"]).sum().reset_index()).set_title('Age vs Suicides')

plt.xticks(rotation=90)
country_suicides = data_n[['country','suicides_no']].groupby(['country']).sum()

country_suicides.plot(kind='bar', figsize=(40,10), fontsize=25)
country_suicides = country_suicides.reset_index().sort_values(by='suicides_no', ascending=False)

top15 = country_suicides[:15]

sns.barplot(x='country', y='suicides_no', data=top15).set_title('countries with most suicides')

plt.xticks(rotation=90)
bottom15 = country_suicides[-15:]

sns.barplot(x='country', y='suicides_no', data=bottom15).set_title('countries with least suicides')

plt.xticks(rotation=90)
data_n[['year','suicides_no']].groupby(['year']).sum().plot()
grid = sns.countplot(x='generation', data=data_n)

grid = plt.setp(grid.get_xticklabels(), rotation=45)
gen_year = data_n[['suicides_no','generation','year']].groupby(['generation','year']).sum().reset_index()

plt.figure(figsize=(25,10))

sns.set(font_scale=1.5)

plt.xticks(rotation=90)

sns.barplot(y='suicides_no', x='year', hue='generation', data=gen_year, palette='deep').set_title('Suicides vs generations per year')
top15data = data_n.loc[data_n['country'].isin(top15.country)]

country_suicides_sex = top15data[['country','suicides_no','sex']].groupby(['country','sex']).sum().reset_index().sort_values(by='suicides_no', ascending=False)

plt.figure(figsize=(25,10))

plt.xticks(rotation=90)

sns.barplot(x='country', y='suicides_no', hue='sex', data=country_suicides_sex).set_title('countries suicides rate w.r.t sex')
bottom15data = data_n.loc[data_n['country'].isin(bottom15.country)]

country_suicides_sex = bottom15data[['country','suicides_no','sex']].groupby(['country','sex']).sum().reset_index().sort_values(by='suicides_no', ascending=False)

plt.figure(figsize=(25,10))

plt.xticks(rotation=90)

sns.barplot(x='country', y='suicides_no', hue='sex', data=country_suicides_sex).set_title('countries suicides rate w.r.t sex')
female_data = data_n.loc[data_n['sex']=='female']

female_suicides = female_data[['country','suicides_no','sex']].groupby(['country','sex']).sum().reset_index().sort_values(by='suicides_no', ascending=False)

plt.figure(figsize=(25,10))

plt.xticks(rotation=90)

sns.barplot(x='country', y='suicides_no', data=female_suicides).set_title('females suicide rate w.r.t country')
f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(data_n.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.figure(figsize=(25,10))

plt.xticks(rotation=90)

sns.barplot(x='country', y='population', hue='sex', data=data_n).set_title('country vs population')
def decade_mapping(data):

    if 1985 <= data <= 1994:

        return "1985-1994"

    elif 1995 <= data <= 2004:

        return "1995-2004"

    else:

        return "2005-2016"

    

data_n.year = data_n.year.apply(decade_mapping)

data_n.head(3)
grid = sns.FacetGrid(data_n, row='generation', col='year', size = 5, aspect=1.5)

grid.map(sns.barplot, 'sex', 'suicides_no', alpha=.5, ci=None)

grid.add_legend()