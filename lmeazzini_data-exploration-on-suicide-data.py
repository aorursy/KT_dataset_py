import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))

df = pd.read_csv('../input/master.csv')
df.sample(5)
df.describe()
df.info()
df.isnull().sum()
df.drop(['country-year', 'HDI for year'], inplace=True, axis = 1)
df = df.rename(columns={'gdp_per_capita ($)': 'gdp_per_capita', ' gdp_for_year ($) ':'gdp_for_year'})
for i, x in enumerate(df['gdp_for_year']):

    df['gdp_for_year'][i] = x.replace(',', '')

    

df['gdp_for_year'] = df['gdp_for_year'].astype('int64')
df['age'].unique()
df['generation'].unique()
Frist_world = ['United States', 'Germany', 'Japan', 'Turkey', 'United Kingdom', 'France', 'Italy', 'South Korea',

              'Spain', 'Canada', 'Australia', 'Netherlands', 'Belgium', 'Greece', 'Portugal', 

              'Sweden', 'Austria', 'Switzerland', 'Israel', 'Singapore', 'Denmark', 'Finland', 'Norway', 'Ireland',

              'New Zeland', 'Slovenia', 'Estonia', 'Cyprus', 'Luxembourg', 'Iceland']



Second_world = ['Russian Federation', 'Ukraine', 'Poland', 'Uzbekistan', 'Romania', 'Kazakhstan', 'Azerbaijan', 'Czech Republic',

               'Hungary', 'Belarus', 'Tajikistan', 'Serbia', 'Bulgaria', 'Slovakia', 'Croatia', 'Maldova', 'Georgia',

               'Bosnia And Herzegovina', 'Albania', 'Armenia', 'Lithuania', 'Latvia', 'Brazil', 'Chile', 'Argentina',

               'China', 'India', 'Bolivia', 'Romenia']
country_world = []

for i in range(len(df)):

    

    if df['country'][i] in Frist_world:

        country_world.append(1)

    elif df['country'][i] in Second_world:

        country_world.append(2)

    else:

        country_world.append(3)



df['country_world'] = country_world
suicides_no_year = []



for y in df['year'].unique():

    suicides_no_year.append(sum(df[df['year'] == y]['suicides_no']))



n_suicides_year = pd.DataFrame(suicides_no_year, columns=['suicides_no_year'])

n_suicides_year['year'] = df['year'].unique()



top_year = n_suicides_year.sort_values('suicides_no_year', ascending=False)['year']

top_suicides = n_suicides_year.sort_values('suicides_no_year', ascending=False)['suicides_no_year']



plt.figure(figsize=(8,5))

plt.xticks(rotation=90)

sns.barplot(x = top_year, y = top_suicides)
suicides_no_age = []



for a in df['age'].unique():

    suicides_no_age.append(sum(df[df['age'] == a]['suicides_no']))



plt.xticks(rotation=30)

sns.barplot(x = df['age'].unique(), y = suicides_no_age)
suicides_no_sex = []



for s in df['sex'].unique():

    suicides_no_sex.append(sum(df[df['sex'] == s]['suicides_no']))



sns.barplot(x = df['sex'].unique(), y = suicides_no_sex)
suicides_no_pais = []

for c in df['country'].unique():

    suicides_no_pais.append(sum(df[df['country'] == c]['suicides_no']))

    

n_suicides_pais = pd.DataFrame(suicides_no_pais, columns=['suicides_no_pais'])

n_suicides_pais['country'] = df['country'].unique()



quant = 15

top_paises = n_suicides_pais.sort_values('suicides_no_pais', ascending=False)['country'][:quant]

top_suicides = n_suicides_pais.sort_values('suicides_no_pais', ascending=False)['suicides_no_pais'][:quant]

sns.barplot(x = top_suicides, y = top_paises)
suicides_no_pais = []

for c in df['country'].unique():

    suicides_no_pais.append(sum(df[df['country'] == c]['suicides/100k pop']))

    

n_suicides_pais = pd.DataFrame(suicides_no_pais, columns=['suicides_no_pais'])

n_suicides_pais['country'] = df['country'].unique()



quant = 15

top_paises = n_suicides_pais.sort_values('suicides_no_pais', ascending=False)['country'][:quant]

top_suicides = n_suicides_pais.sort_values('suicides_no_pais', ascending=False)['suicides_no_pais'][:quant]

sns.barplot(x = top_suicides, y = top_paises)
suicides_no_gen = []

for g in df['generation'].unique():

    suicides_no_gen.append(sum(df[df['generation'] == g]['suicides_no']))



plt.figure(figsize=(8,5))

sns.barplot(x = df['generation'].unique(), y = suicides_no_gen)
suicides_no_world = []

for w in df['country_world'].unique():

    suicides_no_world.append(sum(df[df['country_world'] == w]['suicides_no']))

    

sns.barplot(x = df['country_world'].unique(), y = suicides_no_world)
suicides_no_world = []

for w in df['country_world'].unique():

    suicides_no_world.append(sum(df[df['country_world'] == w]['suicides/100k pop']))

    

sns.barplot(x = df['country_world'].unique(), y = suicides_no_world)
sns.scatterplot(x = 'gdp_for_year', y = 'suicides_no', data = df)
sns.scatterplot(x = 'gdp_per_capita', y = 'suicides_no', data = df)
plt.figure(figsize=(8,7))

sns.heatmap(df.corr(), cmap = 'coolwarm', annot=True)
countries = ['Russian Federation', 'Brazil', 'Poland', 'Italy', 'United States', 'Germany', 'Japan', 'Spain', 'France']

df_filtred = df[[df['country'][i] in countries for i in range(len(df))]]



plt.figure(figsize=(12,6))

sns.boxplot(x = 'suicides/100k pop', y = 'country', data = df_filtred)
import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
cod = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
codes = []

for i in range(len(n_suicides_pais)):

    c = n_suicides_pais['country'][i]

    f = 0

    for j in range(len(cod)):

        if c == cod['COUNTRY'][j]:

            tmp = cod['CODE'][j]

            f = 1

            break

    if f == 0:

        if c == 'Bahamas':

            tmp  = 'BHM'

        elif c == 'Republic of Korea':

            tmp = 'KOR'

        elif c == 'Russian Federation':

            tmp = 'RUS'

        else:

            tmp = 'VC'

    codes.append(tmp)
data = dict(

        type = 'choropleth',

        locations = codes,

        z = n_suicides_pais['suicides_no_pais/100k'],

        text = n_suicides_pais['country'],

        colorbar = {'title' : 'número de suicídios'},

      )
layout = dict(

    title = 'Mapa de calor de suicídios 1985-2016',

    geo = dict(

        showframe = False,

        projection = {'type':'equirectangular'}

    )

)
choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
df_brasil = df[df['country'] == 'Brazil']
df_brasil.drop(['country', 'country_world'], axis = 1, inplace = True)
suicides_no_year = []



for y in df_brasil['year'].unique():

    suicides_no_year.append(sum(df_brasil[df_brasil['year'] == y]['suicides_no']))



n_suicides_year = pd.DataFrame(suicides_no_year, columns=['suicides_no_year'])

n_suicides_year['year'] = df_brasil['year'].unique()



top_year = n_suicides_year.sort_values('suicides_no_year', ascending=False)['year']

top_suicides = n_suicides_year.sort_values('suicides_no_year', ascending=False)['suicides_no_year']



plt.figure(figsize=(8,5))

plt.xticks(rotation=90)

sns.barplot(x = top_year, y = top_suicides)
suicides_no_age = []



for a in df['age'].unique():

    suicides_no_age.append(sum(df_brasil[df_brasil['age'] == a]['suicides_no']))



plt.xticks(rotation=30)

sns.barplot(x = df_brasil['age'].unique(), y = suicides_no_age)
suicides_no_sex = []



for s in df['sex'].unique():

    suicides_no_sex.append(sum(df_brasil[df_brasil['sex'] == s]['suicides_no']))



sns.barplot(x = df_brasil['sex'].unique(), y = suicides_no_sex)
suicides_no_gen = []

for g in df['generation'].unique():

    suicides_no_gen.append(sum(df_brasil[df_brasil['generation'] == g]['suicides_no']))



plt.figure(figsize=(8,5))

sns.barplot(x = df_brasil['generation'].unique(), y = suicides_no_gen)
sns.scatterplot(x = 'gdp_for_year', y = 'suicides_no', data = df_brasil)
sns.scatterplot(x = 'gdp_per_capita', y = 'suicides_no', data = df_brasil)
plt.figure(figsize=(8,7))

sns.heatmap(df_brasil.corr(), cmap = 'coolwarm', annot=True)