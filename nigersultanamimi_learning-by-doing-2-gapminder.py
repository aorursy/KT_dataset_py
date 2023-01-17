import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



sns.set_style("darkgrid")



sns.set(rc={'figure.figsize':(20, 10)})
data = pd.read_csv('../input/gapminder/gapminder.tsv', sep="\t")
data.head()
bd = data.loc[data['country'] == 'Bangladesh']

bd.head()
afg = data.loc[data['country'] == 'Afghanistan']

afg.tail()
asia = data[data['continent'] == 'Asia']

asia.head()
asiaWithSparsePop = asia[asia['pop'] < 1.453083e+07]

sns.boxplot(x="lifeExp", y="country", data=asiaWithSparsePop)
asia['pop'].describe()
bd.head()
bdgdp = sns.boxplot(x='country', y='lifeExp', data=bd)
bd.describe()
bd.shape
bd = bd.assign(gdpQuantiles = lambda x: pd.qcut(x['gdpPercap'] , q=4, labels=['min', 'low', 'medium', 'max']))

sns.boxplot(x='gdpQuantiles', y='year', data=bd)
sns.pairplot(data)
sns.distplot(data['lifeExp'])
data['lifeExp'].describe()
bd['lifeExp']
sns.jointplot(x="gdpPercap", y="pop", data=data)
sns.distplot(bd['lifeExp'])
america = data[data['continent'] == 'Americas']

sns.distplot(america['lifeExp'])
sns.scatterplot(x='lifeExp', y='gdpPercap', size='pop', hue='year', data=data)
data.describe()
gdp = data[data['gdpPercap'] < 9325.462346]

gdp = gdp.assign(popQuartiles = lambda x: pd.qcut(x=x['pop'], q=4))

sns.set(rc={'figure.figsize':(20, 15)})

sns.scatterplot(x='lifeExp', y='gdpPercap', size='popQuartiles', hue='year', data=gdp)
data.groupby('continent').describe()
sns.barplot(y='country', x='pop', data=data[data['continent'] == 'Asia'])
countriesWithLowPopulation = data[data['pop'] < 2.793664e+06 ]

countriesWithLowPopulation.count()
lowPopAsia = countriesWithLowPopulation[countriesWithLowPopulation['continent'] == 'Asia']

sns.barplot(y='country', x='pop', data=lowPopAsia)