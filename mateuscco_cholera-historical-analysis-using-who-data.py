# Standard packages

import json



# Libs to deal with tabular data

import numpy as np

import pandas as pd

pd.options.mode.chained_assignment = None



# Plotting packages

import seaborn as sns

sns.axes_style("darkgrid")

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn')

import plotly.express as px



# To display stuff in notebook

from IPython.display import display, Markdown
df = pd.read_csv('../input/cholera-dataset/data.csv')
df.head()
df.dtypes
df = df.rename(columns={

    'Number of reported cases of cholera':'Cases',

    'Number of reported deaths from cholera':'Deaths',

    'Cholera case fatality rate':'Death rate',

    'WHO Region':'Region'

})
df[df['Cases'].str.contains('[^0-9\.]').astype(bool) & df['Cases'].notnull()]
df[df['Deaths'].str.contains('[^0-9\.]').astype(bool) & df['Deaths'].notnull()]
df[df['Death rate'].str.contains('[^0-9\.]').astype(bool) & df['Death rate'].notnull()]
df.loc[1059, 'Cases'] = '5'

df.loc[1059, 'Deaths'] = '0'

df.loc[1059, 'Death rate'] = '0.0'
for column in ['Cases', 'Deaths', 'Death rate']:

    df[column] = df[column].replace('Unknown', np.nan).str.replace(' ', '')

    df[column] = pd.to_numeric(df[column])
df.isnull().sum()
df[df.isnull().any(1)]
global_year = df.groupby('Year').sum().loc[:, ['Cases', 'Deaths']]



ax = sns.lineplot(data=global_year)

plt.xlabel('Year', fontsize=15)

plt.title('Cholera evolution in the last 70 years', fontsize=16, fontweight='bold')

plt.show()
region_year = df.groupby(['Year', 'Region']).sum()['Cases'].reset_index()

region_year = region_year.pivot(index = 'Year', columns = 'Region', values = 'Cases').fillna(0.0)



region_year.plot.area()

plt.title('Number of reported cases by region', fontsize=16, fontweight='bold')

plt.xlabel('Year', fontsize=15)

plt.show()
codes = pd.read_csv('../input/alpha3-country-codes/alpha3.csv', names=['Code', 'Country']).set_index('Country')



country_year = df.groupby(['Year', 'Country']).sum()['Cases'].reset_index()

country_year = country_year.join(codes, how='left', on='Country')
fig = px.choropleth(

    country_year, 

    locations = "Code",

    color = "Cases",

    hover_name = "Country",

    color_continuous_scale = px.colors.sequential.Plasma,

    animation_frame = 'Year',

    animation_group = 'Country',

    range_color = [0, 100000]

)

fig.show()
n_countries_year = df.groupby(['Year', 'Region']).count()['Country'].rename('Number of countries').reset_index()

n_countries_year = n_countries_year.pivot(index = 'Year', columns = 'Region', values = 'Number of countries')

n_countries_year = n_countries_year.fillna(0.0)



n_countries_year.plot.area()

plt.title('Number of countries reporting data by region', fontsize=16, fontweight='bold')

plt.xlabel('Year', fontsize=15)

plt.show()
country_agg = df.groupby('Country').sum().loc[:, ['Cases', 'Deaths']]

country_agg['Death rate'] = country_agg['Deaths'] * 100 / country_agg['Cases']

country_agg = country_agg.sort_values('Cases', ascending=False).head(10)



country_agg.loc[:, ['Deaths', 'Cases']].iloc[::-1].plot(kind='barh', figsize=(8,6), rot=0, colormap='coolwarm_r')

plt.title('Contries most affected by cholera', fontsize=16, fontweight='bold')

plt.ylabel('Country', fontsize=15)

plt.show()
country_agg = df.groupby('Country').sum().loc[:, ['Cases', 'Deaths']]

country_agg['Death rate'] = country_agg['Deaths'] * 100 / country_agg['Cases']

country_agg = country_agg.sort_values('Death rate', ascending=False).head(10)



sns.barplot(country_agg['Death rate'].values, country_agg.index, palette='Reds_r')

plt.title('Highest death rates overall', fontsize=16, fontweight='bold')

plt.xlabel('Percentage (%)', fontsize=15)

plt.ylabel('Country', fontsize=15)

plt.show()
last_decade = df.loc[df['Year'] >= 2007, :]

country_agg = last_decade.groupby('Country').sum().loc[:, ['Cases', 'Deaths']]

country_agg['Death rate'] = country_agg['Deaths'] * 100 / country_agg['Cases']

country_agg = country_agg.sort_values('Death rate', ascending=False).head(10)



sns.barplot(country_agg['Death rate'].values, country_agg.index, palette='Reds_r')

plt.title('Highest death rates since 2007', fontsize=16, fontweight='bold')

plt.xlabel('Percentage (%)', fontsize=15)

plt.ylabel('Country', fontsize=15)

plt.show()
hdi = pd.read_csv('../input/human-development/human_development.csv')

hdi['Country'] = hdi['Country'].replace({

    'Tanzania (United Republic of)':'United Republic of Tanzania',

    'United Kingdom':'United Kingdom of Great Britain and Northern Ireland',

    'Congo (Democratic Republic of the)':'Democratic Republic of the Congo',

    'United States':'United States of America'

})

hdi['Gross National Income (GNI) per Capita'] = pd.to_numeric(hdi['Gross National Income (GNI) per Capita'].str.replace(',', ''))

hdi = hdi.set_index('Country')

hdi = hdi.iloc[:, 1:6]



data_2015 = df.loc[df['Year'] == 2015, ['Country', 'Cases', 'Deaths', 'Death rate']].set_index('Country')

data_2015 = data_2015.join(hdi, how='left')



print('In 2015, {} countries reported cholera data'.format(data_2015.shape[0]))



correlations = data_2015.corr(method='pearson')

correlations = correlations.iloc[:3, 3:]



sns.heatmap(data=correlations.T, annot=True, color=sns.color_palette("coolwarm", 7))

plt.title('Correlation between development indexes and cholera metrics', fontsize=16, fontweight='bold')

plt.xticks(rotation=0) 

plt.yticks(rotation=0) 

plt.show()
fig, ax = plt.subplots(5, 3, figsize=(12,16))



for i, col_i in enumerate([

    'Human Development Index (HDI)', 'Life Expectancy at Birth',

    'Expected Years of Education', 'Mean Years of Education',

    'Gross National Income (GNI) per Capita']):

    for j, col_j in enumerate(['Cases', 'Deaths', 'Death rate']):        

        sns.regplot(x = data_2015[col_j], y = data_2015[col_i], ci=None, ax=ax[i, j])

        if(i != 4):

            #ax[i, j].get_xaxis().set_ticks([])

            ax[i, j].set_xlabel('')

        if(j != 0):

            #ax[i, j].get_yaxis().set_ticks([])

            ax[i, j].set_ylabel('')

        



plt.suptitle('Relationship between indexes', fontsize=16, fontweight='bold')

plt.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
health_indexes = pd.read_csv('../input/sanitation-and-water-global-indexes/SanitationAndWaterGlobalIndexes/indexes.csv')



health_indexes = health_indexes.loc[health_indexes['Year'] == 2015, [

    'Country',

    'Population using at least basic drinking-water services (%) - Total',

    'Population using safely managed drinking-water services (%) - Total',

    'Population using at least basic sanitation services (%) - Total',

    'Population using safely managed sanitation services (%) - Total',

    'Population with basic handwashing facilities at home (%) - Total',

    'Population practising open defecation (%) - Total'

]]



health_indexes = health_indexes.set_index('Country').rename(columns={

    'Population using at least basic drinking-water services (%) - Total':'Basic drinking-water services (%)',

    'Population using safely managed drinking-water services (%) - Total':'Safe drinking-water services (%)',

    'Population using at least basic sanitation services (%) - Total':'Basic sanitation services (%)',

    'Population using safely managed sanitation services (%) - Total':'Safe sanitation services (%)',

    'Population with basic handwashing facilities at home (%) - Total':'Basic handwashing facilities (%)',

    'Population practising open defecation (%) - Total':'Open defecation (%)'

})
print('In 2015, {} countries at least one of the above indexes.'.format(health_indexes.shape[0]))



health_indexes.isnull().sum() * 100 / health_indexes.shape[0]
health_data_2015 = data_2015.loc[:, ['Cases', 'Deaths', 'Death rate']].join(health_indexes, how='left')



correlations = health_data_2015.corr(method='pearson')

correlations = correlations.iloc[:3, 3:]



sns.heatmap(data=correlations.T, annot=True, color=sns.color_palette("coolwarm", 7))

plt.title('Correlation between public health indexes and cholera metrics', fontsize=16, fontweight='bold')

plt.xticks(rotation=0) 

plt.yticks(rotation=0) 

plt.show()