import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns
!ls '../input/'
sns.set(style="whitegrid")
suicides_df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

suicides_df.head()
suicides_df.info()
suicides_df.isnull().sum()
suicides_df.shape
suicides_df.drop(['HDI for year', 'country-year'], axis=1, inplace=True)
suicides_df.rename(columns=lambda x: x.strip(), inplace=True)

suicides_df.columns
suic_total = suicides_df[['country','suicides_no']].groupby('country').sum()

suic_total = suic_total.reset_index()

suic_total.sort_values(by=['suicides_no'], ascending=False, inplace=True)

suic_total.head()
fig, ax = plt.subplots(figsize=(20,5))

sns.barplot(y='suicides_no', x='country', data=suic_total.head(10), ax=ax, palette=sns.color_palette('cubehelix'))

plt.title('Countries with highest number of suicides in the last 3 decades (1985 - 2016)')

plt.ylabel('count')
suic_rate = suicides_df[['country','suicides/100k pop']].groupby('country').sum()

suic_rate = suic_rate.reset_index()

suic_rate.sort_values(by=['suicides/100k pop'], ascending=False, inplace=True)

suic_rate.head()
fig, ax = plt.subplots(figsize=(20,5))

sns.barplot(y='suicides/100k pop', x='country', data=suic_rate.head(10), ax=ax, palette=sns.color_palette('cubehelix'))

plt.title('Countries with highest rate (suicides per 100k population) of suicides in the last 3 decades (1985 - 2016)')

plt.ylabel('count')
top_countries = suic_rate.head(10)['country']

top_countries.reset_index(drop=True)
suic_gender = suicides_df.loc[suicides_df['country'].isin(top_countries)]

suic_gender = suic_gender[['country', 'sex', 'suicides/100k pop']].groupby(['country', 'sex']).sum()

suic_gender = suic_gender.reset_index()

suic_gender.sort_values(by=['suicides/100k pop'], ascending=False, inplace=True)

suic_gender.head()
fig, ax = plt.subplots(figsize=(20,5))

sns.barplot(y='suicides/100k pop', x='country', hue='sex', data=suic_gender, ax=ax, palette=sns.color_palette('cubehelix', 2))

plt.title('Countries with highest number of suicides in the last 3 decades (1985 - 2016)')

plt.ylabel('suicide count')
suic_year = suicides_df.loc[suicides_df['country'].isin(top_countries)]

suic_year = suic_year[['country', 'sex', 'suicides/100k pop', 'year']].groupby(['country', 'sex', 'year']).sum()

suic_year = suic_year.reset_index()

suic_year.sort_values(by=['year', 'suicides/100k pop'], ascending=(True, False), inplace=True)

suic_year.head()
bps = sns.FacetGrid(suic_year, col='country', col_wrap=1, height=3, aspect=3.5, hue='sex', 

                    palette=sns.color_palette('cubehelix', 2))

bps.map(sns.barplot, 'year', 'suicides/100k pop', hue_order='sex')

bps.add_legend()

bps.set_ylabels('suicide rate')

bps.set_xticklabels(suic_year['year'].unique())



axs = bps.axes.flatten()

for i in range(len(suic_year['country'].unique())):

    axs[i].set_title(suic_year['country'].unique()[i])
suic_age = suicides_df.loc[suicides_df['country'].isin(top_countries)]

suic_age = suic_age[['country', 'age', 'suicides/100k pop']].groupby(['country', 'age']).sum()

suic_age = suic_age.reset_index()

suic_age['age'] = suic_age['age'].map({'5-14 years': 1, '15-24 years': 2, '25-34 years': 3, 

                        '35-54 years': 4, '55-74 years': 5, '75+ years': 6})

suic_age.sort_values(by=['suicides/100k pop', 'age'], ascending=(False, True), inplace=True)

suic_age.head()
fig, ax = plt.subplots(figsize=(20,5))

colors = sns.color_palette('cubehelix', 6)

sns.barplot(y='suicides/100k pop', x='country', hue='age', data=suic_age, ax=ax, palette=colors)

plt.title('Countries with highest number of suicides in the last 3 decades (1985 - 2016) categorised by age')

plt.ylabel('suicide rate')

plt.legend(labels=['5-14 yo', '15-24 yo', '25-34 yo', '35-54 yo', '55-74 yo', '75+ yo'])



legends = ax.get_legend()

for i in range(len(legends.legendHandles)):

    legends.legendHandles[i].set_color(colors[i])

suic_gen = suicides_df.loc[suicides_df['country'].isin(top_countries)]

suic_gen = suic_gen[['country', 'generation', 'suicides/100k pop']].groupby(['country', 'generation']).sum()

suic_gen = suic_gen.reset_index()

suic_gen['generation'] = suic_gen['generation'].map({'G.I. Generation': 1, 'Silent': 2, 'Boomers': 3, 

                        'Generation X': 4, 'Millennials': 5, 'Generation Z': 6})

suic_gen.sort_values(by=['suicides/100k pop', 'generation'], ascending=(False, True), inplace=True)

suic_gen.head()
generations = ['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millennials', 'Generation Z']
fig, ax = plt.subplots(figsize=(20,5))

colors = sns.color_palette('cubehelix', 6)

sns.barplot(y='suicides/100k pop', x='country', hue='generation', data=suic_gen, ax=ax, palette=colors)

plt.title('Countries with highest number of suicides in the last 3 decades (1985 - 2016) categorised by generation')

plt.ylabel('suicide rate')



boxes = [item for item in ax.get_children() if isinstance(item, matplotlib.patches.Rectangle)][:-1]

legend_patches = [matplotlib.patches.Patch(color=color, label=label) for

                  color, label in zip([item.get_facecolor() for item in boxes], generations)]

plt.legend(handles=legend_patches)



legends = ax.get_legend()

for i in range(len(legends.legendHandles)):

    legends.legendHandles[i].set_color(colors[i])
gdp = suicides_df[['country', 'gdp_per_capita ($)']].groupby('country').sum()

gdp = gdp.reset_index()

gdp.sort_values(by=['gdp_per_capita ($)'], ascending=False, inplace=True)

gdp.head()
top_countries = top_countries.tolist()
top_gdp = gdp.head(25)

clrs = ['red' if (row['country'] in top_countries) else 'grey' for _, row in top_gdp.iterrows()]
fig, ax = plt.subplots(figsize=(20,7))

bp = sns.barplot(y='gdp_per_capita ($)', x='country', data=top_gdp, ax=ax, palette=clrs)

bp.set_xticklabels(bp.get_xticklabels(), rotation=90)

plt.title('Countries with highest GDP per capita in the last 3 decades (1985 - 2016)')

plt.ylabel('gdp')