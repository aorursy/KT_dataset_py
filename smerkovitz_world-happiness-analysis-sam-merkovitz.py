import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

sns.set(font='Franklin Gothic Book')

sns.set_context("notebook", rc={"font.size":16,

                                "axes.titlesize":20,

                                "axes.labelsize":18})
hp_2015 = pd.read_csv('/kaggle/input/world-happiness-analysis/2015.csv')

hp_2015['Year'] = '2015'

hp_2016 = pd.read_csv('/kaggle/input/world-happiness-analysis/2016.csv')

hp_2016['Year'] = '2016'

hp_2017 = pd.read_csv('/kaggle/input/world-happiness-analysis/2017.csv')

hp_2017['Year'] = '2017'

hp_2018 = pd.read_csv('/kaggle/input/world-happiness-analysis/2018.csv')

hp_2018['Year'] = '2018'

hp_2019 = pd.read_csv('/kaggle/input/world-happiness-analysis/2019.csv')

hp_2019['Year'] = '2019'



hp_2015.head()
plt.figure(figsize=(10,10))

sns.scatterplot('Economy (GDP per Capita)', 'Happiness Score', data=hp_2015, hue='Region')
correlations = hp_2015.corr()

correlations
corr_2016 = hp_2016[['Happiness Score', 'Economy (GDP per Capita)', 'Health (Life Expectancy)', 'Freedom']].corr()

corr_2016
corr_2017 = hp_2017[['Happiness Score', 'Economy (GDP per Capita)', 'Health (Life Expectancy)', 'Freedom']].corr()

corr_2017
corr_2018 = hp_2018[['Happiness Score', 'Economy (GDP per Capita)', 'Health (Life Expectancy)', 'Freedom']].corr()

corr_2018
corr_2019 = hp_2019[['Happiness Score', 'Economy (GDP per Capita)', 'Health (Life Expectancy)', 'Freedom']].corr()

corr_2019
def year_over_year(measures, dfs, country='United States'):

  country_over_years = pd.DataFrame()

  for df in dfs:

    temp = df[df['Country'] == country]

    temp = temp[measures]

    country_over_years = pd.concat((country_over_years, temp))

  return country_over_years



usa = year_over_year(['Year', 'Happiness Score', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)'], [hp_2015, hp_2016, hp_2017, hp_2018, hp_2019])

usa
usa['Happiness Change'] = usa['Happiness Score'].apply(lambda x: x - 6)

usa
plt.figure(figsize=(10,10))

df = usa



sns.lineplot(x='Year',y='Happiness Change', data=df, color='black', label='Happiness')

sns.lineplot(x='Year',y='Economy (GDP per Capita)', data=df, color='blue', label='Economy')

sns.lineplot(x='Year',y='Family', data=df, color='red', label='Family')

sns.lineplot(x='Year',y='Health (Life Expectancy)', data=df, color='green', label='Health')

plt.legend()

plt.xlabel('Year')

plt.ylabel('Score')

plt.title('US Happiness Score Year over Year')
fin = year_over_year(['Year', 'Happiness Score', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)'], [hp_2015, hp_2016, hp_2017, hp_2018, hp_2019], country='Finland')

fin['Happiness Change'] = fin['Happiness Score'].apply(lambda x: x - 6)

fin
plt.figure(figsize=(10,10))

df = fin



sns.lineplot(x='Year',y='Happiness Change', data=df, color='black', label='Happiness')

sns.lineplot(x='Year',y='Economy (GDP per Capita)', data=df, color='blue', label='Economy')

sns.lineplot(x='Year',y='Family', data=df, color='red', label='Family')

sns.lineplot(x='Year',y='Health (Life Expectancy)', data=df, color='green', label='Health')

plt.legend()

plt.xlabel('Year')

plt.ylabel('Score')

plt.title('Finland Happiness Score Year over Year')
usa_corr = usa.corr()

usa_corr
fin_corr = fin.corr()

fin_corr
usa_2 = year_over_year(['Year', 'Happiness Score', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',

       'Generosity'], [hp_2015, hp_2016, hp_2017, hp_2018, hp_2019])

usa_2_corr = usa_2.corr()

usa_2_corr
fin_2 = year_over_year(['Year', 'Happiness Score', 'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',

       'Generosity'], [hp_2015, hp_2016, hp_2017, hp_2018, hp_2019], country='Finland')

fin_2_corr = fin_2.corr()

fin_2_corr
gdp = pd.read_csv('/kaggle/input/world-happiness-analysis/GDP.csv')

gdp.head()
hp_2019_gdp = pd.merge(hp_2019, gdp, left_on='Country', right_on='country').drop(columns=['country'])

hp_2019_gdp.head()
import math

hp_2019_gdp['LogGDP'] = hp_2019_gdp['gdpPerCapita'].apply(lambda x: math.log(x))

hp_2019_gdp_corr = hp_2019_gdp[['Happiness Score', 'Economy (GDP per Capita)', 'LogGDP', 'rank', 'gdpPerCapita', 'pop']].corr()

hp_2019_gdp_corr
plt.figure(figsize=(10,10))

df = hp_2019_gdp



sns.regplot(x='LogGDP',y='Economy (GDP per Capita)', data=df, color='black')

sns.scatterplot(x='LogGDP',y='Economy (GDP per Capita)', data=df[df['Country'] == 'Finland'], color='orange', label='Finland')

sns.scatterplot(x='LogGDP',y='Economy (GDP per Capita)', data=df[df['Country'] == 'United States'], color='red', label='United States')

sns.scatterplot(x='LogGDP',y='Economy (GDP per Capita)', data=df[df['Country'] == 'Mexico'], color='green', label='Mexico')



plt.legend()

plt.title('LogGDP vs. Perceived Importance')

plt.xlabel('Actual GDP (Log)')

plt.ylabel('Perceived Importance to Happiness')
plt.figure(figsize=(10,10))

df = hp_2019_gdp



sns.scatterplot(x='gdpPerCapita',y='Economy (GDP per Capita)', data=df, color='black')

sns.scatterplot(x='gdpPerCapita',y='Economy (GDP per Capita)', data=df[df['Country'] == 'Finland'], color='orange', label='Finland')

sns.scatterplot(x='gdpPerCapita',y='Economy (GDP per Capita)', data=df[df['Country'] == 'United States'], color='red', label='United States')

sns.scatterplot(x='gdpPerCapita',y='Economy (GDP per Capita)', data=df[df['Country'] == 'Mexico'], color='green', label='Mexico')



plt.legend()

plt.title('GDP vs. Perceived Importance')

plt.xlabel('Actual GDP')

plt.ylabel('Perceived Importance to Happiness')