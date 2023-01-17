import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gridspec

import matplotlib.style as style



style.use('ggplot')



ID = 'id'

TARGET = 'target'

NFOLDS = 7

SEED = 18

NROWS = None

DATA_DIR = '../input/suicide-rates-overview-1985-to-2016'



DATA_FILE = f'{DATA_DIR}/master.csv'
df = pd.read_csv(DATA_FILE)

df.sample(8)
def tableSummary(df):

    from scipy import stats

    print(f'Dataset Shape: {df.shape}')

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary
generations_order = ['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials', 'Generation Z']
df.rename(columns={

    ' gdp_for_year ($) ': 'gdp_for_year',

    'gdp_per_capita ($)': 'gdp_per_capita',

    'suicides/100k pop': 'suicides/100k',

}, inplace=True)
# Drop country-year because is useless

# Drop HDI because too many missing values

df = df.drop(['country-year', 'HDI for year'], axis=1)



df['country'] = df['country'].astype('category')

df['sex'] = df['sex'].astype('category')

df['age'] = df['age'].astype('category')

df['generation'] = df['generation'].astype('category')



# Convert GDP to numerical value

df['gdp_for_year'] = df['gdp_for_year'].apply(lambda x: int(x.replace(',', ''))).astype('int64')
tableSummary(df)
# Data cleaning took from https://www.kaggle.com/fredzanella/should-we-care-about-money-an-eda-on-suicide

# Thanks fredzanella for the amazing work!



agg_dict = { 'country':'nunique', 'age':'nunique',

             'population':'sum', 'suicides_no':'sum',

             'suicides/100k':'mean' }



both_ends = df.query('year < 1988 | year > 2013')



both_ends = both_ends[['year', 'country',

                       'age', 'population',

                       'suicides_no',

                       'suicides/100k']].groupby('year').agg(agg_dict)

both_ends
# Remove 2016 for data inconsistency

df = df.query('year != 2016')
aggr = { 'population':'sum', 'suicides_no':'sum' }



df_group_year = df.groupby(['year']).agg(aggr).reset_index()

df_group_year['suicides/100k'] = 100000 * df_group_year['suicides_no'] / df_group_year['population']

df_group_year.head()
fig = plt.figure(figsize = (16, 8))

sns.barplot(x='year', y='suicides_no', data=df_group_year, palette='rocket')

fig.suptitle('Suicides rate by Year', fontsize=18)

plt.show()
fig, ax = plt.subplots(figsize=(16,8))

df_group_year.plot(x='year', y='suicides/100k', ax=ax)

fig.suptitle('Suicides rate over 100k by Year', fontsize=18)

plt.show()
fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))

sns.lineplot(data=df_group_year, y='population', x='year', ax=ax1, label='Population')

ax1.set_ylim(1e9, 2.6e9)

ax1.legend(bbox_to_anchor=(1.112, 0.1))



ax2 = plt.twinx()

sns.lineplot(data=df_group_year, y='suicides_no', x='year', ax=ax2, color='C3', label='Suicides')

ax2.set_ylim(1e5, 2.6e5)

ax2.legend(bbox_to_anchor=(1.1, 0.21))



plt.title('Population vs number of suicides by year', fontsize=18)

plt.show()
df_group_sex = df.groupby(['sex']).agg({ 'suicides_no': 'sum' }).reset_index()

df_group_sex.head()
fig = plt.figure(figsize = (8, 6))

sns.barplot(x='sex', y='suicides_no', data=df_group_sex, palette='rocket')

fig.suptitle('Suicides rate by Sex', fontsize=18)

plt.show()
df_group_year_sex = df.groupby(['year', 'sex']).agg({ 'suicides_no': 'sum' }).reset_index()

df_group_year_sex.head()
fig = plt.figure(figsize = (16, 8))

sns.barplot(x='year', y='suicides_no', data=df_group_year_sex, hue='sex', palette='rocket')

fig.suptitle('Suicides rate by Year', fontsize=18)

plt.show()
df_jp = df[df['country'] == 'Japan']

df_fr = df[df['country'] == 'France']
df_mix1 = df_jp.groupby(['sex']).agg({ 'suicides_no': 'sum' }).reset_index()

df_mix1['country'] = 'Japan'

df_mix2 = df_fr.groupby(['sex']).agg({ 'suicides_no': 'sum' }).reset_index()

df_mix2['country'] = 'France'



df_mix = pd.concat([df_mix1, df_mix2])

df_mix
fig = plt.figure(figsize = (8, 6))

sns.barplot(x='country', y='suicides_no', data=df_mix, hue='sex', palette='rocket')

fig.suptitle('Suicides rate by Sex - Japan vs France', fontsize=18)

plt.show()
df_group_country = df.groupby(['country']).agg({ 'suicides_no': 'sum' }).reset_index()

df_group_country.head()
top_countries = df_group_country.sort_values('suicides_no', ascending=False)[:20]



fig = plt.figure(figsize = (16, 8))

g = sns.barplot(x='country', y='suicides_no', data=top_countries, order=top_countries['country'], palette='rocket')

fig.suptitle('Suicides rate by Country - top 20', fontsize=18)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.show()
df_group_country_p = df.groupby(['country']).agg({ 'suicides/100k': 'mean' }).reset_index()

df_group_country_p.head()
top_countries = df_group_country_p.sort_values('suicides/100k', ascending=False)[:20]



fig = plt.figure(figsize = (16, 8))

g = sns.barplot(x='country', y='suicides/100k', data=top_countries, order=top_countries['country'], palette='rocket')

fig.suptitle('Suicides rate over 100k citizen - top 20', fontsize=18)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.show()
bottom_countries = df_group_country_p.sort_values('suicides/100k', ascending=True)

bottom_countries.head(10)
country_occ = []



for country in df_group_country_p['country']:

    years = df[df['country'] == country]['year'].nunique()

    country_occ.append({

        'country': country,

        'year_no': years

    })



df_year_country = pd.DataFrame(country_occ)

df_year_country.sample(8)
print('Mean occurrences for each country:', df_year_country['year_no'].mean())
df_year_country.sort_values('year_no', ascending=True).query('year_no < 15')
df_gcp = df_group_country_p

df_ycfiltered = df_year_country.query('year_no >= 15')

bottom_countries = df_gcp[df_gcp['country'].isin(df_ycfiltered['country'])].sort_values('suicides/100k', ascending=True)[:15]



fig = plt.figure(figsize = (16, 8))

g = sns.barplot(x='country', y='suicides/100k', data=bottom_countries, order=bottom_countries['country'], palette='rocket')

fig.suptitle('Suicides rate over 100k citizen - bottom 15 (filtered)', fontsize=18)

g.set_xticklabels(g.get_xticklabels(), rotation=45)

plt.show()
df_group_year_gen = df.groupby(['year', 'generation']).agg({ 'suicides_no': 'sum' }).reset_index()

df_group_year_gen['suicides_no'].fillna(0, inplace=True)

df_group_year_gen['suicides_no'] = df_group_year_gen['suicides_no'].astype('int64')

df_group_year_gen.head(10)
grid = gridspec.GridSpec(30, 2)

fig = plt.figure(figsize=(20, 150))

fig.subplots_adjust(hspace=0.4, wspace=0.3)



min_year = min(df['year'])

max_year = max(df['year'])



for n, year in enumerate(range(min_year, max_year + 1)):

    df_y = df_group_year_gen[df_group_year_gen['year'] == year]

    ax = plt.subplot(grid[n])

    sns.barplot(x='generation', y='suicides_no', data=df_y, palette='rocket', order=generations_order)

    ax.set_title(f'Suicides rate by Generation - {year}', fontsize=16)



plt.show()
df_group_gen = df.groupby(['generation']).agg({ 'suicides_no': 'sum' }).reset_index()

df_group_gen
rocketPalette = sns.color_palette('rocket', n_colors=8)



patches, texts, autotexts = plt.pie(df_group_gen['suicides_no'],

                                    colors=rocketPalette,

                                    labels=df_group_gen['generation'],

                                    autopct='%1.1f%%',

                                    startangle=90)

plt.title('Division of suicides by Generation', fontsize=20, y=1.05)

for text in texts:

    text.set_fontsize(12)

for autotext in autotexts:

    autotext.set_color('white')

    autotext.set_fontsize(12)

plt.axis('equal')

plt.tight_layout()

plt.show()
df_group_age = df.groupby(['age']).agg({ 'suicides_no': 'sum' }).reset_index()

df_group_age
rocketPalette = sns.color_palette('rocket', n_colors=8)



patches, texts, autotexts = plt.pie(df_group_age['suicides_no'],

                                    colors=rocketPalette,

                                    labels=df_group_age['age'],

                                    autopct='%1.1f%%',

                                    startangle=90)

plt.title('Division of suicides by Age', fontsize=20, y=1.05)

for text in texts:

    text.set_fontsize(12)

for autotext in autotexts:

    autotext.set_color('white')

    autotext.set_fontsize(12)

plt.axis('equal')

plt.tight_layout()

plt.show()
sns.jointplot(x='suicides_no', y='gdp_per_capita', data=df)

plt.suptitle('Relation GDP per capita with Suicides number', fontsize=18, y=1.05)

plt.show()
df_ru = df[df['country'] == 'Russian Federation']

df_us = df[df['country'] == 'United States']
df_ru_gdp = df_ru.groupby(['year']).agg({ 'suicides_no': 'sum', 'gdp_for_year': 'mean' }).reset_index()



sns.jointplot(x='suicides_no', y='gdp_for_year', data=df_ru_gdp, kind='reg')

plt.suptitle('Relation GDP for year with Suicides number - Russian Federation', fontsize=18, y=1.05)

plt.show()
df_us_gdp = df_us.groupby(['year']).agg({ 'suicides_no': 'sum', 'gdp_for_year': 'mean' }).reset_index()



sns.jointplot(x='suicides_no', y='gdp_for_year', data=df_us_gdp, kind='reg')

plt.suptitle('Relation GDP for year with Suicides number - USA', fontsize=18, y=1.05)

plt.show()